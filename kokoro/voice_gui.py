"""
Voice Creation GUI for Kokoro TTS

Interactive slider-based interface for creating custom voices using PCA.
"""

import torch
import numpy as np
import gradio as gr
from typing import Optional, Tuple, List
from pathlib import Path
import json
import soundfile as sf


class VoiceCreationGUI:
    """
    Gradio-based GUI for interactive voice creation with PCA sliders.

    Features:
    - Sliders for each PCA component (up to 99.9% variance coverage)
    - 5x exaggeration support
    - Real-time voice preview
    - Save custom voices
    - Load existing voices for editing
    """

    def __init__(
        self,
        pca_system,
        pipeline=None,
        max_exaggeration: float = 5.0,
        test_text: str = "Hello, this is a test of my new voice."
    ):
        """
        Initialize the Voice Creation GUI.

        Args:
            pca_system: VoicePCASystem instance (must be fitted)
            pipeline: Optional KPipeline instance for audio preview
            max_exaggeration: Maximum slider range (default 5x)
            test_text: Default test text for voice preview
        """
        self.pca_system = pca_system
        self.pipeline = pipeline
        self.max_exaggeration = max_exaggeration
        self.test_text = test_text

        # Validate PCA system
        if pca_system.pca_model is None or pca_system.centroid is None:
            raise ValueError("PCA system must be fitted before use")

        # Calculate bounds with exaggeration
        if pca_system.pca_bounds is None:
            pca_system.calculate_pca_bounds(max_exaggeration=max_exaggeration)
        elif pca_system.pca_bounds.get('exaggeration', 1.0) != max_exaggeration:
            pca_system.calculate_pca_bounds(max_exaggeration=max_exaggeration)

        self.n_components = pca_system.n_components
        self.pca_bounds = pca_system.pca_bounds

        # Current voice state
        self.current_pca_coords = np.zeros(self.n_components)
        self.current_voice_tensor = None

    def create_interface(self, share: bool = False, server_port: int = 7860) -> gr.Blocks:
        """
        Create the Gradio interface.

        Args:
            share: Whether to create a public link
            server_port: Port to run the server on

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Kokoro Voice Creator") as interface:
            gr.Markdown("# 🎙️ Kokoro Voice Creator")
            gr.Markdown(
                f"Create custom voices by adjusting PCA components. "
                f"Using **{self.n_components} components** "
                f"(covering {self.pca_system.explained_variance_ratio.sum()*100:.1f}% variance) "
                f"with **{self.max_exaggeration}x exaggeration**."
            )

            with gr.Row():
                # Left column: Controls
                with gr.Column(scale=1):
                    gr.Markdown("## 🎚️ Voice Controls")

                    # Voice name input
                    voice_name = gr.Textbox(
                        label="Voice Name",
                        value="custom_voice",
                        placeholder="Enter a name for your voice"
                    )

                    # Preset buttons
                    with gr.Row():
                        reset_btn = gr.Button("🔄 Reset to Centroid", size="sm")
                        random_btn = gr.Button("🎲 Random Voice", size="sm")
                        save_btn = gr.Button("💾 Save Voice", size="sm")

                    # Load existing voice
                    with gr.Row():
                        voice_dropdown = gr.Dropdown(
                            choices=self.pca_system.voice_names,
                            label="Load Existing Voice",
                            value=None
                        )
                        load_btn = gr.Button("📂 Load", size="sm")

                    # PCA component sliders
                    gr.Markdown("### PCA Components")
                    gr.Markdown(f"*Adjust sliders to modify voice characteristics*")

                    sliders = []
                    for i in range(min(self.n_components, 50)):  # Limit to first 50 for UI
                        var_pct = self.pca_system.explained_variance_ratio[i] * 100
                        slider = gr.Slider(
                            minimum=float(self.pca_bounds['min'][i]),
                            maximum=float(self.pca_bounds['max'][i]),
                            value=0.0,
                            step=0.01,
                            label=f"PC{i+1} ({var_pct:.2f}% var)",
                            interactive=True
                        )
                        sliders.append(slider)

                    # If more than 50 components, show message
                    if self.n_components > 50:
                        gr.Markdown(
                            f"*Showing first 50 of {self.n_components} components. "
                            f"Remaining components are set to 0.*"
                        )

                # Right column: Preview and Info
                with gr.Column(scale=1):
                    gr.Markdown("## 🎵 Voice Preview")

                    # Test text input
                    test_text_input = gr.Textbox(
                        label="Test Text",
                        value=self.test_text,
                        lines=3
                    )

                    # Preview button
                    preview_btn = gr.Button("▶️ Generate Audio Preview", variant="primary")

                    # Audio output
                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="numpy",
                        interactive=False
                    )

                    # Voice statistics
                    gr.Markdown("### Voice Statistics")
                    stats_output = gr.JSON(label="Current Voice Stats")

                    # Save status
                    save_status = gr.Textbox(label="Status", interactive=False)

            # Event handlers
            def update_voice_from_sliders(*slider_values):
                """Update voice when sliders change."""
                # Convert slider values to PCA coords
                pca_coords = np.zeros(self.n_components)
                pca_coords[:len(slider_values)] = slider_values

                self.current_pca_coords = pca_coords

                # Transform to voice tensor
                voice_tensor = self.pca_system.inverse_pca_transform(pca_coords)
                self.current_voice_tensor = voice_tensor

                # Calculate statistics
                stats = self._calculate_voice_stats(pca_coords, voice_tensor)

                return stats

            def reset_to_centroid():
                """Reset all sliders to 0 (centroid)."""
                zero_coords = [0.0] * min(self.n_components, 50)
                stats = update_voice_from_sliders(*zero_coords)
                return zero_coords + [stats]

            def generate_random():
                """Generate random voice."""
                from .synthetic_voices import SyntheticVoiceGenerator

                generator = SyntheticVoiceGenerator(
                    self.pca_system,
                    max_exaggeration=self.max_exaggeration,
                    flatten_factor=0.3
                )

                voice_data = generator.generate_random_voice()
                pca_coords = voice_data['pca_coords']

                self.current_pca_coords = pca_coords
                self.current_voice_tensor = voice_data['voice_tensor']

                # Update sliders (only first 50)
                slider_values = pca_coords[:min(self.n_components, 50)].tolist()
                stats = self._calculate_voice_stats(pca_coords, voice_data['voice_tensor'])

                return slider_values + [voice_data['name'], stats]

            def load_existing_voice(voice_name):
                """Load an existing voice and update sliders."""
                if voice_name is None:
                    return [gr.update()] * (min(self.n_components, 50) + 1)

                voice_tensor = self.pca_system.get_voice_by_name(voice_name)
                pca_coords = self.pca_system.voice_to_pca(voice_tensor)

                self.current_pca_coords = pca_coords
                self.current_voice_tensor = voice_tensor

                slider_values = pca_coords[:min(self.n_components, 50)].tolist()
                stats = self._calculate_voice_stats(pca_coords, voice_tensor)

                return slider_values + [stats]

            def save_voice(name):
                """Save current voice to disk."""
                if self.current_voice_tensor is None:
                    return "Error: No voice to save"

                save_dir = Path("voices/custom")
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save voice tensor
                voice_path = save_dir / f"{name}.pt"
                torch.save(self.current_voice_tensor, voice_path)

                # Save metadata
                metadata = {
                    'name': name,
                    'pca_coords': self.current_pca_coords.tolist(),
                    'shape': list(self.current_voice_tensor.shape),
                    'exaggeration': self.max_exaggeration
                }

                metadata_path = save_dir / f"{name}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                return f"✅ Voice saved to {voice_path}"

            def generate_preview(test_text):
                """Generate audio preview of current voice."""
                if self.pipeline is None:
                    return None

                if self.current_voice_tensor is None:
                    # Use centroid
                    self.current_voice_tensor = self.pca_system.centroid

                try:
                    # Generate audio
                    audio_generator = self.pipeline(
                        test_text,
                        voice=self.current_voice_tensor,
                        speed=1.0
                    )

                    # Collect audio chunks
                    audio_chunks = []
                    for chunk in audio_generator:
                        audio_chunks.append(chunk)

                    # Concatenate audio
                    audio = np.concatenate(audio_chunks)

                    # Return as (sample_rate, audio_data)
                    return (24000, audio)

                except Exception as e:
                    print(f"Error generating preview: {e}")
                    return None

            # Wire up events
            # Slider changes update stats
            for slider in sliders:
                slider.change(
                    fn=update_voice_from_sliders,
                    inputs=sliders,
                    outputs=[stats_output]
                )

            # Reset button
            reset_btn.click(
                fn=reset_to_centroid,
                inputs=[],
                outputs=sliders + [stats_output]
            )

            # Random button
            random_btn.click(
                fn=generate_random,
                inputs=[],
                outputs=sliders + [voice_name, stats_output]
            )

            # Load button
            load_btn.click(
                fn=load_existing_voice,
                inputs=[voice_dropdown],
                outputs=sliders + [stats_output]
            )

            # Save button
            save_btn.click(
                fn=save_voice,
                inputs=[voice_name],
                outputs=[save_status]
            )

            # Preview button
            preview_btn.click(
                fn=generate_preview,
                inputs=[test_text_input],
                outputs=[audio_output]
            )

            gr.Markdown("""
            ---
            ### Tips:
            - **PC1-PC5** typically control the most significant voice characteristics
            - Extreme values create more exaggerated, unique voices
            - Try loading existing voices and tweaking them slightly
            - Use the preview button to hear your voice before saving
            """)

        self.interface = interface
        return interface

    def _calculate_voice_stats(
        self,
        pca_coords: np.ndarray,
        voice_tensor: torch.Tensor
    ) -> dict:
        """Calculate statistics for the current voice."""
        # L2 distance from centroid
        centered = (voice_tensor - self.pca_system.centroid).cpu().numpy().flatten()
        distance_from_centroid = np.linalg.norm(centered)

        # PCA coordinate statistics
        pca_magnitude = np.linalg.norm(pca_coords)
        pca_active_components = np.sum(np.abs(pca_coords) > 0.01)

        # Voice tensor statistics
        voice_np = voice_tensor.cpu().numpy()

        stats = {
            "Distance from Centroid": f"{distance_from_centroid:.4f}",
            "PCA Magnitude": f"{pca_magnitude:.4f}",
            "Active Components": int(pca_active_components),
            "Voice Mean": f"{voice_np.mean():.6f}",
            "Voice Std": f"{voice_np.std():.6f}",
            "Voice Range": f"[{voice_np.min():.4f}, {voice_np.max():.4f}]",
            "Shape": f"{list(voice_tensor.shape)}"
        }

        return stats

    def launch(
        self,
        share: bool = False,
        server_port: int = 7860,
        **kwargs
    ):
        """
        Launch the Gradio interface.

        Args:
            share: Create a public link
            server_port: Port to run on
            **kwargs: Additional arguments for gr.Blocks.launch()
        """
        if not hasattr(self, 'interface'):
            self.create_interface()

        self.interface.launch(
            share=share,
            server_port=server_port,
            **kwargs
        )


def launch_voice_gui(
    pca_system_path: Optional[str] = None,
    voices_dir: Optional[str] = None,
    share: bool = False,
    server_port: int = 7860
):
    """
    Convenience function to launch the GUI.

    Args:
        pca_system_path: Path to saved PCA system (will create if None)
        voices_dir: Path to voices directory
        share: Create public link
        server_port: Server port
    """
    from .voice_pca_system import VoicePCASystem

    # Load or create PCA system
    if pca_system_path and Path(pca_system_path).exists():
        print(f"Loading PCA system from {pca_system_path}")
        pca_system = VoicePCASystem(voices_dir=voices_dir)
        pca_system.load_system(pca_system_path)
    else:
        print("Creating new PCA system...")
        pca_system = VoicePCASystem(
            voices_dir=voices_dir,
            variance_coverage=0.999
        )
        pca_system.load_all_voices()
        pca_system.calculate_centroid()
        pca_system.fit_pca()

        # Save for future use
        if pca_system_path:
            pca_system.save_system(pca_system_path)

    # Try to load pipeline for audio preview
    try:
        from .pipeline import KPipeline
        pipeline = KPipeline(lang_code='a')
        print("Pipeline loaded for audio preview")
    except Exception as e:
        print(f"Could not load pipeline: {e}")
        pipeline = None

    # Create and launch GUI
    gui = VoiceCreationGUI(
        pca_system=pca_system,
        pipeline=pipeline,
        max_exaggeration=5.0
    )

    gui.create_interface()
    gui.launch(share=share, server_port=server_port)
