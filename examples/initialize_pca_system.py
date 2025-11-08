#!/usr/bin/env python3
"""
Initialize PCA System for Voice Manipulation

This script loads all voices, calculates the centroid, fits PCA,
and saves the system for future use.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.voice_pca_system import VoicePCASystem


def main():
    """Initialize and save the PCA system."""

    print("=" * 60)
    print("Initializing Kokoro Voice PCA System")
    print("=" * 60)
    print()

    # Configuration
    voices_dir = Path(__file__).parent.parent / "voices"
    save_dir = voices_dir / "pca_system"
    variance_coverage = 0.999  # 99.9% variance coverage

    print(f"Configuration:")
    print(f"  Voices directory: {voices_dir}")
    print(f"  Save directory: {save_dir}")
    print(f"  Variance coverage: {variance_coverage * 100:.1f}%")
    print()

    # Initialize PCA system
    print("Step 1: Initializing PCA system...")
    pca_system = VoicePCASystem(
        voices_dir=str(voices_dir),
        variance_coverage=variance_coverage
    )
    print("✓ System initialized")
    print()

    # Load all voices
    print("Step 2: Loading all voices...")
    pca_system.load_all_voices()
    print(f"✓ Loaded {len(pca_system.voice_data)} voices")
    print()

    # Calculate centroid
    print("Step 3: Calculating centroid...")
    centroid = pca_system.calculate_centroid()
    print("✓ Centroid calculated")
    print()

    # Fit PCA
    print("Step 4: Fitting PCA model...")
    pca_model = pca_system.fit_pca()
    print("✓ PCA fitted")
    print(f"  Components: {pca_system.n_components}")
    print(f"  Actual variance: {pca_system.explained_variance_ratio.sum() * 100:.2f}%")
    print()

    # Calculate bounds for GUI (5x exaggeration)
    print("Step 5: Calculating PCA bounds (5x exaggeration)...")
    bounds_5x = pca_system.calculate_pca_bounds(max_exaggeration=5.0)
    print("✓ Bounds calculated")
    print()

    # Save system
    print("Step 6: Saving PCA system...")
    pca_system.save_system(str(save_dir))
    print("✓ System saved")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Voices loaded: {len(pca_system.voice_data)}")
    print(f"Centroid shape: {centroid.shape}")
    print(f"PCA components: {pca_system.n_components}")
    print(f"Variance coverage: {pca_system.explained_variance_ratio.sum() * 100:.2f}%")
    print(f"Top 5 components: {pca_system.explained_variance_ratio[:5]}")
    print()
    print(f"PCA system saved to: {save_dir}")
    print()
    print("You can now:")
    print("  1. Generate synthetic voices: python examples/generate_synthetic_voices.py")
    print("  2. Launch the GUI: python examples/launch_voice_gui.py")
    print()


if __name__ == "__main__":
    main()
