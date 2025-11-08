#!/usr/bin/env python3
"""
Launch Voice Creation GUI

This script launches the interactive Gradio GUI for creating
custom voices using PCA sliders.
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.voice_gui import launch_voice_gui


def main():
    """Launch the voice creation GUI."""

    parser = argparse.ArgumentParser(description="Launch Voice Creation GUI")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    parser.add_argument(
        "--pca-system",
        type=str,
        default=None,
        help="Path to PCA system directory (default: auto-detect)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Launching Kokoro Voice Creation GUI")
    print("=" * 60)
    print()

    # Auto-detect PCA system path
    if args.pca_system is None:
        pca_system_path = Path(__file__).parent.parent / "voices" / "pca_system"
    else:
        pca_system_path = Path(args.pca_system)

    # Check if PCA system exists
    if not pca_system_path.exists():
        print("ERROR: PCA system not found!")
        print(f"Expected location: {pca_system_path}")
        print()
        print("Please run: python examples/initialize_pca_system.py")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  PCA system: {pca_system_path}")
    print(f"  Server port: {args.port}")
    print(f"  Public share: {args.share}")
    print()

    print("Starting GUI...")
    print()

    # Launch GUI
    voices_dir = Path(__file__).parent.parent / "voices"
    launch_voice_gui(
        pca_system_path=str(pca_system_path),
        voices_dir=str(voices_dir),
        share=args.share,
        server_port=args.port
    )


if __name__ == "__main__":
    main()
