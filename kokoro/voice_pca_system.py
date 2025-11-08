"""
Voice PCA System for Kokoro TTS

This module provides PCA-based voice manipulation capabilities:
- Centroid calculation across all voices
- PCA decomposition with configurable variance coverage
- Inverse PCA for voice reconstruction from PCA coordinates
- Exaggeration support (up to 5x for GUI, 2x for random generation)
- Distribution-aware random voice generation
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from huggingface_hub import hf_hub_download
import json


class VoicePCASystem:
    """
    PCA-based voice manipulation system for Kokoro TTS.

    Handles centroid calculation, PCA decomposition, inverse transforms,
    and voice generation with exaggeration support.
    """

    def __init__(
        self,
        voices_dir: Optional[str] = None,
        repo_id: str = "hexgrad/Kokoro-82M",
        variance_coverage: float = 0.999,
        device: str = "cpu"
    ):
        """
        Initialize the Voice PCA System.

        Args:
            voices_dir: Directory containing voice .pt files (optional)
            repo_id: HuggingFace repo ID for downloading voices
            variance_coverage: Target variance coverage (e.g., 0.999 for 99.9%)
            device: Device to use ('cpu' or 'cuda')
        """
        self.voices_dir = Path(voices_dir) if voices_dir else None
        self.repo_id = repo_id
        self.variance_coverage = variance_coverage
        self.device = device

        # Storage for loaded data
        self.voice_data: Dict[str, torch.Tensor] = {}
        self.centroid: Optional[torch.Tensor] = None
        self.pca_model: Optional[PCA] = None
        self.pca_bounds: Optional[Dict[str, np.ndarray]] = None
        self.voice_names: List[str] = []

        # Statistics
        self.n_components: Optional[int] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None

    def load_all_voices(self, voice_list: Optional[List[str]] = None) -> None:
        """
        Load all voice embeddings from disk or HuggingFace.

        Args:
            voice_list: Optional list of voice names to load. If None, loads default set.
        """
        if voice_list is None:
            # Default voice list (all major voices)
            voice_list = [
                # American Female
                'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica',
                'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
                # American Male
                'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
                'am_michael', 'am_onyx', 'am_puck', 'am_santa',
                # British Female
                'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
                # British Male
                'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
                # Other languages
                'ef_arlet', 'ef_dora', 'em_danel', 'em_ramos',
                'ff_adele', 'hf_diya', 'hm_prabhu',
                'if_chiara', 'im_raffa',
                'jf_himari', 'jf_nana', 'jm_eiji', 'jm_kaito',
                'pf_ana', 'pm_dinis',
                'zf_meimei'
            ]

        self.voice_names = voice_list
        print(f"Loading {len(voice_list)} voices...")

        for i, voice_name in enumerate(voice_list):
            try:
                voice_tensor = self._load_single_voice(voice_name)
                self.voice_data[voice_name] = voice_tensor
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{len(voice_list)} voices")
            except Exception as e:
                print(f"  Warning: Could not load {voice_name}: {e}")

        print(f"Successfully loaded {len(self.voice_data)} voices")

    def _load_single_voice(self, voice_name: str) -> torch.Tensor:
        """Load a single voice embedding."""
        # Try local directory first
        if self.voices_dir and (self.voices_dir / f"{voice_name}.pt").exists():
            voice_path = self.voices_dir / f"{voice_name}.pt"
        else:
            # Download from HuggingFace
            voice_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"voices/{voice_name}.pt"
            )

        voice_tensor = torch.load(voice_path, map_location=self.device, weights_only=True)
        return voice_tensor

    def calculate_centroid(self) -> torch.Tensor:
        """
        Calculate the centroid (mean) of all loaded voices.

        Returns:
            Centroid tensor of shape [seq_len, 256]
        """
        if not self.voice_data:
            raise ValueError("No voices loaded. Call load_all_voices() first.")

        # Find the most common sequence length
        seq_lengths = [v.shape[0] for v in self.voice_data.values()]
        # Use median sequence length as reference
        ref_seq_len = int(np.median(seq_lengths))

        # Collect all embeddings, interpolating to reference length
        all_embeddings = []
        for voice_name, voice_tensor in self.voice_data.items():
            # Interpolate to reference sequence length if needed
            if voice_tensor.shape[0] != ref_seq_len:
                # Interpolate along sequence dimension
                voice_interp = torch.nn.functional.interpolate(
                    voice_tensor.T.unsqueeze(0),  # [1, 256, seq_len]
                    size=ref_seq_len,
                    mode='linear',
                    align_corners=True
                ).squeeze(0).T  # [ref_seq_len, 256]
            else:
                voice_interp = voice_tensor

            all_embeddings.append(voice_interp)

        # Stack and compute mean
        all_embeddings = torch.stack(all_embeddings, dim=0)  # [n_voices, seq_len, 256]
        self.centroid = all_embeddings.mean(dim=0)  # [seq_len, 256]

        print(f"Centroid calculated with shape: {self.centroid.shape}")
        print(f"  Mean: {self.centroid.mean().item():.6f}")
        print(f"  Std:  {self.centroid.std().item():.6f}")
        print(f"  Min:  {self.centroid.min().item():.6f}")
        print(f"  Max:  {self.centroid.max().item():.6f}")

        return self.centroid

    def fit_pca(self) -> PCA:
        """
        Fit PCA model on all loaded voices (centered around centroid).

        Returns:
            Fitted PCA model
        """
        if self.centroid is None:
            raise ValueError("Centroid not calculated. Call calculate_centroid() first.")

        ref_seq_len = self.centroid.shape[0]

        # Prepare centered data matrix
        centered_voices = []
        for voice_name, voice_tensor in self.voice_data.items():
            # Interpolate to reference length
            if voice_tensor.shape[0] != ref_seq_len:
                voice_interp = torch.nn.functional.interpolate(
                    voice_tensor.T.unsqueeze(0),
                    size=ref_seq_len,
                    mode='linear',
                    align_corners=True
                ).squeeze(0).T
            else:
                voice_interp = voice_tensor

            # Center around centroid
            centered = voice_interp - self.centroid
            centered_voices.append(centered.cpu().numpy())

        # Stack into data matrix [n_voices, seq_len * 256]
        X = np.array([v.flatten() for v in centered_voices])

        print(f"Fitting PCA on data matrix of shape: {X.shape}")

        # Fit PCA with target variance coverage
        self.pca_model = PCA(n_components=self.variance_coverage, svd_solver='full')
        self.pca_model.fit(X)

        self.n_components = self.pca_model.n_components_
        self.explained_variance_ratio = self.pca_model.explained_variance_ratio_

        print(f"PCA fitted with {self.n_components} components")
        print(f"  Variance coverage: {self.pca_model.explained_variance_ratio_.sum():.4f}")
        print(f"  Top 5 component variances: {self.explained_variance_ratio[:5]}")

        return self.pca_model

    def calculate_pca_bounds(self, max_exaggeration: float = 5.0) -> Dict[str, np.ndarray]:
        """
        Calculate bounds for each PCA component based on actual voice data.

        Args:
            max_exaggeration: Maximum exaggeration factor (e.g., 5.0 for 5x)

        Returns:
            Dictionary with 'min' and 'max' arrays for each component
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")

        ref_seq_len = self.centroid.shape[0]

        # Transform all voices to PCA space
        pca_coords = []
        for voice_name, voice_tensor in self.voice_data.items():
            # Interpolate and center
            if voice_tensor.shape[0] != ref_seq_len:
                voice_interp = torch.nn.functional.interpolate(
                    voice_tensor.T.unsqueeze(0),
                    size=ref_seq_len,
                    mode='linear',
                    align_corners=True
                ).squeeze(0).T
            else:
                voice_interp = voice_tensor

            centered = (voice_interp - self.centroid).cpu().numpy().flatten()
            pca_coord = self.pca_model.transform([centered])[0]
            pca_coords.append(pca_coord)

        pca_coords = np.array(pca_coords)  # [n_voices, n_components]

        # Calculate bounds with exaggeration
        mins = pca_coords.min(axis=0) * max_exaggeration
        maxs = pca_coords.max(axis=0) * max_exaggeration

        self.pca_bounds = {
            'min': mins,
            'max': maxs,
            'data_min': pca_coords.min(axis=0),  # Actual data bounds
            'data_max': pca_coords.max(axis=0),
            'mean': pca_coords.mean(axis=0),
            'std': pca_coords.std(axis=0),
            'exaggeration': max_exaggeration
        }

        print(f"PCA bounds calculated with {max_exaggeration}x exaggeration")
        print(f"  Component ranges (first 5):")
        for i in range(min(5, self.n_components)):
            print(f"    PC{i+1}: [{mins[i]:.4f}, {maxs[i]:.4f}]")

        return self.pca_bounds

    def inverse_pca_transform(
        self,
        pca_coords: np.ndarray,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Transform PCA coordinates back to voice embedding space.

        Args:
            pca_coords: PCA coordinates array of shape [n_components]
            seq_len: Target sequence length (uses centroid length if None)

        Returns:
            Voice embedding tensor of shape [seq_len, 256]
        """
        if self.pca_model is None or self.centroid is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")

        # Inverse transform to flattened centered voice
        centered_flat = self.pca_model.inverse_transform([pca_coords])[0]

        # Reshape to [seq_len, 256]
        ref_seq_len = self.centroid.shape[0]
        centered_voice = torch.from_numpy(centered_flat.reshape(ref_seq_len, 256)).float()

        # Add centroid back
        voice = centered_voice + self.centroid

        # Interpolate to target sequence length if needed
        if seq_len is not None and seq_len != ref_seq_len:
            voice = torch.nn.functional.interpolate(
                voice.T.unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=True
            ).squeeze(0).T

        return voice

    def voice_to_pca(self, voice: torch.Tensor) -> np.ndarray:
        """
        Transform a voice embedding to PCA coordinates.

        Args:
            voice: Voice embedding tensor of shape [seq_len, 256]

        Returns:
            PCA coordinates array of shape [n_components]
        """
        if self.pca_model is None or self.centroid is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")

        ref_seq_len = self.centroid.shape[0]

        # Interpolate to reference length
        if voice.shape[0] != ref_seq_len:
            voice = torch.nn.functional.interpolate(
                voice.T.unsqueeze(0),
                size=ref_seq_len,
                mode='linear',
                align_corners=True
            ).squeeze(0).T

        # Center and flatten
        centered = (voice - self.centroid).cpu().numpy().flatten()

        # Transform to PCA space
        pca_coords = self.pca_model.transform([centered])[0]

        return pca_coords

    def get_voice_by_name(self, voice_name: str) -> torch.Tensor:
        """Get a loaded voice by name."""
        if voice_name not in self.voice_data:
            raise ValueError(f"Voice '{voice_name}' not loaded")
        return self.voice_data[voice_name]

    def save_system(self, save_dir: str) -> None:
        """
        Save the PCA system to disk.

        Args:
            save_dir: Directory to save the system files
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save centroid
        if self.centroid is not None:
            torch.save(self.centroid, save_path / "centroid.pt")

        # Save PCA model
        if self.pca_model is not None:
            import pickle
            with open(save_path / "pca_model.pkl", 'wb') as f:
                pickle.dump(self.pca_model, f)

        # Save bounds
        if self.pca_bounds is not None:
            np.savez(save_path / "pca_bounds.npz", **self.pca_bounds)

        # Save metadata
        metadata = {
            'variance_coverage': self.variance_coverage,
            'n_components': self.n_components,
            'voice_names': self.voice_names,
            'centroid_shape': list(self.centroid.shape) if self.centroid is not None else None,
        }
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"PCA system saved to {save_path}")

    def load_system(self, load_dir: str) -> None:
        """
        Load a saved PCA system from disk.

        Args:
            load_dir: Directory containing saved system files
        """
        load_path = Path(load_dir)

        # Load centroid
        centroid_path = load_path / "centroid.pt"
        if centroid_path.exists():
            self.centroid = torch.load(centroid_path, map_location=self.device, weights_only=True)

        # Load PCA model
        pca_path = load_path / "pca_model.pkl"
        if pca_path.exists():
            import pickle
            with open(pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            self.n_components = self.pca_model.n_components_
            self.explained_variance_ratio = self.pca_model.explained_variance_ratio_

        # Load bounds
        bounds_path = load_path / "pca_bounds.npz"
        if bounds_path.exists():
            self.pca_bounds = dict(np.load(bounds_path))

        # Load metadata
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.variance_coverage = metadata.get('variance_coverage', 0.999)
            self.voice_names = metadata.get('voice_names', [])

        print(f"PCA system loaded from {load_path}")
        if self.n_components:
            print(f"  Components: {self.n_components}")
            print(f"  Variance coverage: {self.pca_model.explained_variance_ratio_.sum():.4f}")
