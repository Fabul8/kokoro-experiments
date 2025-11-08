#!/usr/bin/env python3
"""
Comprehensive Voice Manipulation Demo

This script demonstrates all features of the voice manipulation system:
1. Loading and analyzing voices
2. PCA transformation and inverse
3. Creating custom voices from PCA coordinates
4. Generating synthetic voices
5. Saving and loading custom voices
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.voice_pca_system import VoicePCASystem
from kokoro.synthetic_voices import SyntheticVoiceGenerator


def print_section(title):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print()


def demo_pca_system():
    """Demonstrate PCA system functionality."""

    print_section("1. INITIALIZING PCA SYSTEM")

    # Initialize
    pca_system = VoicePCASystem(
        voices_dir="voices",
        variance_coverage=0.999
    )

    # Load voices
    print("Loading voices...")
    pca_system.load_all_voices()
    print(f"✓ Loaded {len(pca_system.voice_data)} voices")

    # Calculate centroid
    print("\nCalculating centroid...")
    centroid = pca_system.calculate_centroid()
    print(f"✓ Centroid shape: {centroid.shape}")

    # Fit PCA
    print("\nFitting PCA...")
    pca_system.fit_pca()
    print(f"✓ Components: {pca_system.n_components}")
    print(f"✓ Variance coverage: {pca_system.explained_variance_ratio.sum() * 100:.2f}%")

    # Calculate bounds
    print("\nCalculating PCA bounds (5x exaggeration)...")
    pca_system.calculate_pca_bounds(max_exaggeration=5.0)
    print(f"✓ Bounds calculated")

    return pca_system


def demo_voice_transformation(pca_system):
    """Demonstrate voice to PCA and back."""

    print_section("2. VOICE TRANSFORMATION (Voice ↔ PCA)")

    # Get a voice
    voice_name = "af_bella"
    print(f"Loading voice: {voice_name}")
    voice = pca_system.get_voice_by_name(voice_name)
    print(f"  Original shape: {voice.shape}")

    # Transform to PCA
    print("\nTransforming to PCA space...")
    pca_coords = pca_system.voice_to_pca(voice)
    print(f"✓ PCA coordinates shape: {pca_coords.shape}")
    print(f"  PCA magnitude: {np.linalg.norm(pca_coords):.4f}")
    print(f"  Top 5 components: {pca_coords[:5]}")

    # Transform back
    print("\nTransforming back to voice space...")
    reconstructed = pca_system.inverse_pca_transform(pca_coords)
    print(f"✓ Reconstructed shape: {reconstructed.shape}")

    # Calculate reconstruction error
    if voice.shape == reconstructed.shape:
        error = torch.mean((voice - reconstructed) ** 2).item()
        print(f"  Reconstruction MSE: {error:.6f}")
    else:
        print(f"  Shapes differ (original interpolated), comparing interpolated version...")
        # The reconstruction uses reference length, so compare at that length
        ref_len = pca_system.centroid.shape[0]
        voice_interp = torch.nn.functional.interpolate(
            voice.T.unsqueeze(0), size=ref_len, mode='linear', align_corners=True
        ).squeeze(0).T
        error = torch.mean((voice_interp - reconstructed) ** 2).item()
        print(f"  Reconstruction MSE: {error:.6f}")


def demo_custom_voice_creation(pca_system):
    """Demonstrate creating custom voices from PCA coordinates."""

    print_section("3. CREATING CUSTOM VOICES FROM PCA")

    # Example 1: Modify specific components
    print("Example 1: Boosting PC1 and PC2")
    pca_coords = np.zeros(pca_system.n_components)
    pca_coords[0] = 2.0   # Boost PC1
    pca_coords[1] = -1.5  # Reduce PC2
    pca_coords[2] = 1.0   # Boost PC3

    voice = pca_system.inverse_pca_transform(pca_coords)
    print(f"✓ Created voice with shape: {voice.shape}")
    print(f"  PCA coords: {pca_coords[:5]}")
    print(f"  Voice stats: mean={voice.mean():.4f}, std={voice.std():.4f}")

    # Example 2: Blend between two voices
    print("\nExample 2: Blending two voices")
    voice1_name = "af_bella"
    voice2_name = "bf_alice"

    voice1 = pca_system.get_voice_by_name(voice1_name)
    voice2 = pca_system.get_voice_by_name(voice2_name)

    pca1 = pca_system.voice_to_pca(voice1)
    pca2 = pca_system.voice_to_pca(voice2)

    # 70% voice1, 30% voice2
    blended_pca = 0.7 * pca1 + 0.3 * pca2
    blended_voice = pca_system.inverse_pca_transform(blended_pca)

    print(f"✓ Blended {voice1_name} (70%) + {voice2_name} (30%)")
    print(f"  Blended voice shape: {blended_voice.shape}")
    print(f"  Distance from voice1: {np.linalg.norm(pca1 - blended_pca):.4f}")
    print(f"  Distance from voice2: {np.linalg.norm(pca2 - blended_pca):.4f}")

    # Example 3: Exaggerate a voice
    print("\nExample 3: Exaggerating a voice (2x)")
    original_pca = pca_system.voice_to_pca(voice1)
    exaggerated_pca = original_pca * 2.0  # 2x exaggeration

    exaggerated_voice = pca_system.inverse_pca_transform(exaggerated_pca)
    print(f"✓ Exaggerated {voice1_name} by 2x")
    print(f"  Original magnitude: {np.linalg.norm(original_pca):.4f}")
    print(f"  Exaggerated magnitude: {np.linalg.norm(exaggerated_pca):.4f}")


def demo_synthetic_generation(pca_system):
    """Demonstrate synthetic voice generation."""

    print_section("4. SYNTHETIC VOICE GENERATION")

    # Initialize generator
    print("Initializing synthetic voice generator...")
    generator = SyntheticVoiceGenerator(
        pca_system=pca_system,
        max_exaggeration=2.0,
        flatten_factor=0.3,
        seed=42  # For reproducibility
    )
    print("✓ Generator initialized")

    # Generate single voice
    print("\nGenerating single random voice...")
    voice1 = generator.generate_random_voice(distribution='normal')
    print(f"✓ Generated: {voice1['name']}")
    print(f"  Shape: {voice1['voice_tensor'].shape}")
    print(f"  PCA magnitude: {np.linalg.norm(voice1['pca_coords']):.4f}")
    print(f"  Top 5 PCA: {voice1['pca_coords'][:5]}")

    # Generate batch
    print("\nGenerating batch of 5 random voices...")
    voices = generator.generate_batch(n_voices=5)
    print(f"✓ Generated {len(voices)} voices:")
    for v in voices:
        print(f"  - {v['name']}: magnitude={np.linalg.norm(v['pca_coords']):.4f}")

    # Show sampling statistics
    print("\nSampling statistics:")
    stats = generator.get_sampling_statistics()
    print(f"  Components: {stats['n_components']}")
    print(f"  Max exaggeration: {stats['max_exaggeration']}x")
    print(f"  Flatten factor: {stats['flatten_factor']}")
    print(f"  Sampling std range: {stats['sampling_std_range']}")

    return voices


def demo_save_load(pca_system, voices):
    """Demonstrate saving and loading."""

    print_section("5. SAVING AND LOADING")

    save_dir = Path("voices/demo_output")

    # Save PCA system
    print("Saving PCA system...")
    pca_system.save_system("voices/demo_pca_system")
    print("✓ PCA system saved")

    # Save synthetic voices
    print("\nSaving synthetic voices...")
    from kokoro.synthetic_voices import SyntheticVoiceGenerator
    generator = SyntheticVoiceGenerator(pca_system, max_exaggeration=2.0)
    generator.save_voices(voices, save_dir)
    print(f"✓ Voices saved to {save_dir}")

    # Load PCA system
    print("\nLoading PCA system...")
    new_pca_system = VoicePCASystem()
    new_pca_system.load_system("voices/demo_pca_system")
    print("✓ PCA system loaded")
    print(f"  Components: {new_pca_system.n_components}")

    # Load a saved voice
    print("\nLoading saved voice...")
    voice_files = list(save_dir.glob("*.pt"))
    if voice_files:
        loaded_voice = torch.load(voice_files[0], weights_only=True)
        print(f"✓ Loaded {voice_files[0].stem}")
        print(f"  Shape: {loaded_voice.shape}")


def demo_statistics(pca_system):
    """Show detailed statistics."""

    print_section("6. STATISTICS AND ANALYSIS")

    print("PCA Components Analysis:")
    print(f"  Total components: {pca_system.n_components}")
    print(f"  Total variance: {pca_system.explained_variance_ratio.sum() * 100:.2f}%")
    print()

    print("Top 10 components:")
    for i in range(min(10, pca_system.n_components)):
        var_pct = pca_system.explained_variance_ratio[i] * 100
        cumsum = pca_system.explained_variance_ratio[:i+1].sum() * 100
        print(f"  PC{i+1:2d}: {var_pct:6.2f}% (cumulative: {cumsum:6.2f}%)")

    print("\nPCA Bounds (5x exaggeration):")
    for i in range(min(5, pca_system.n_components)):
        min_val = pca_system.pca_bounds['min'][i]
        max_val = pca_system.pca_bounds['max'][i]
        data_min = pca_system.pca_bounds['data_min'][i]
        data_max = pca_system.pca_bounds['data_max'][i]
        print(f"  PC{i+1}: [{min_val:7.3f}, {max_val:7.3f}]  "
              f"(data: [{data_min:7.3f}, {data_max:7.3f}])")

    print("\nCentroid statistics:")
    centroid = pca_system.centroid
    print(f"  Shape: {centroid.shape}")
    print(f"  Mean: {centroid.mean().item():.6f}")
    print(f"  Std: {centroid.std().item():.6f}")
    print(f"  Range: [{centroid.min().item():.6f}, {centroid.max().item():.6f}]")


def main():
    """Run the complete demo."""

    print("=" * 70)
    print(" KOKORO VOICE MANIPULATION SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 70)

    # Run all demos
    pca_system = demo_pca_system()
    demo_voice_transformation(pca_system)
    demo_custom_voice_creation(pca_system)
    voices = demo_synthetic_generation(pca_system)
    demo_save_load(pca_system, voices)
    demo_statistics(pca_system)

    print_section("DEMO COMPLETE")
    print("✓ All demonstrations completed successfully!")
    print()
    print("Next steps:")
    print("  1. Run: python examples/initialize_pca_system.py")
    print("  2. Run: python examples/generate_synthetic_voices.py")
    print("  3. Run: python examples/launch_voice_gui.py")
    print()


if __name__ == "__main__":
    main()
