#!/usr/bin/env python3
"""
Generate Synthetic Voices

This script generates random synthetic voices using the PCA system
with distribution-aware sampling and flattening.
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.voice_pca_system import VoicePCASystem
from kokoro.synthetic_voices import SyntheticVoiceGenerator, VoiceNameGenerator


def main():
    """Generate synthetic voices."""

    parser = argparse.ArgumentParser(description="Generate synthetic voices")
    parser.add_argument(
        "--n-voices",
        type=int,
        default=10,
        help="Number of voices to generate (default: 10)"
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=2.0,
        help="Max exaggeration factor (default: 2.0)"
    )
    parser.add_argument(
        "--flatten",
        type=float,
        default=0.3,
        help="Distribution flattening factor 0-1 (default: 0.3)"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="normal",
        choices=["normal", "uniform", "truncated_normal"],
        help="Sampling distribution (default: normal)"
    )
    parser.add_argument(
        "--tail-reduction",
        type=float,
        default=0.7,
        help="Tail reduction factor 0-1 (default: 0.7)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="voices/synthetic",
        help="Output directory (default: voices/synthetic)"
    )
    parser.add_argument(
        "--name-style",
        type=str,
        default="random",
        choices=["random", "fantasy", "tech", "nature"],
        help="Voice name style (default: random)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating Synthetic Voices")
    print("=" * 60)
    print()

    # Configuration
    pca_system_dir = Path(__file__).parent.parent / "voices" / "pca_system"
    output_dir = Path(__file__).parent.parent / args.output_dir

    print(f"Configuration:")
    print(f"  PCA system: {pca_system_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of voices: {args.n_voices}")
    print(f"  Max exaggeration: {args.exaggeration}x")
    print(f"  Flatten factor: {args.flatten}")
    print(f"  Distribution: {args.distribution}")
    print(f"  Tail reduction: {args.tail_reduction}")
    print(f"  Name style: {args.name_style}")
    print()

    # Load PCA system
    print("Step 1: Loading PCA system...")
    if not pca_system_dir.exists():
        print("ERROR: PCA system not found!")
        print("Please run: python examples/initialize_pca_system.py")
        sys.exit(1)

    pca_system = VoicePCASystem()
    pca_system.load_system(str(pca_system_dir))
    print("✓ PCA system loaded")
    print(f"  Components: {pca_system.n_components}")
    print()

    # Initialize generator
    print("Step 2: Initializing synthetic voice generator...")
    generator = SyntheticVoiceGenerator(
        pca_system=pca_system,
        max_exaggeration=args.exaggeration,
        flatten_factor=args.flatten
    )
    print("✓ Generator initialized")
    print()

    # Generate voices
    print("Step 3: Generating synthetic voices...")
    voices = generator.generate_batch(
        n_voices=args.n_voices,
        distribution=args.distribution,
        tail_reduction=args.tail_reduction,
        save_dir=str(output_dir)
    )
    print("✓ Voices generated and saved")
    print()

    # Display results
    print("=" * 60)
    print("Generated Voices")
    print("=" * 60)
    print()

    for i, voice in enumerate(voices[:5], 1):  # Show first 5
        print(f"{i}. {voice['name']}")
        print(f"   Shape: {voice['voice_tensor'].shape}")
        print(f"   PCA magnitude: {np.linalg.norm(voice['pca_coords']):.4f}")
        print(f"   Distribution: {voice['distribution']}")
        print()

    if len(voices) > 5:
        print(f"... and {len(voices) - 5} more voices")
        print()

    # Statistics
    print("=" * 60)
    print("Sampling Statistics")
    print("=" * 60)
    print()

    stats = generator.get_sampling_statistics()
    print(f"Components: {stats['n_components']}")
    print(f"Max exaggeration: {stats['max_exaggeration']}x")
    print(f"Flatten factor: {stats['flatten_factor']}")
    print(f"Sampling std range: {stats['sampling_std_range']}")
    print(f"Component weights range: {stats['component_weights_range']}")
    print()

    print(f"All voices saved to: {output_dir}")
    print()


if __name__ == "__main__":
    import numpy as np
    main()
