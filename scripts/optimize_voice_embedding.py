#!/usr/bin/env python3
"""
Voice Embedding Gradient Descent Optimizer - Main Script

This script optimizes Kokoro TTS voice embeddings to match a target audio file
using gradient descent on the most variable dimensions in PCA space.

Usage:
    python scripts/optimize_voice_embedding.py \\
        --target-audio path/to/target.wav \\
        --output-dir voice_optimizations/experiment1 \\
        --text "Hello, this is a test." \\
        --iterations 1000 \\
        --learning-rate 0.01

Features:
- Uses WavLM for SOTA voice similarity embeddings
- Analyzes all voice_pt files to find most variable dimensions
- Optimizes in PCA-reduced space (top 256 components)
- Uses median centroid as starting point
- GPU-accelerated throughout
- Finite difference gradient estimation for black-box optimization
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro import KPipeline
from kokoro.voice_embedding_optimizer import (
    OptimizerConfig,
    create_optimizer
)
import torch
import warnings


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Kokoro voice embedding to match target audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--target-audio",
        type=str,
        required=True,
        help="Path to target audio file (WAV, MP3, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save optimized voice and checkpoints"
    )

    # Voice analysis
    parser.add_argument(
        "--voices-dir",
        type=str,
        default="voices",
        help="Directory containing voice .pt files (optional)"
    )
    parser.add_argument(
        "--n-variable-dims",
        type=int,
        default=256,
        help="Number of most variable dimensions to use"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=256,
        help="Number of PCA components for dimensionality reduction"
    )

    # Optimization
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help="Optimizer type"
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="cosine",
        choices=["cosine", "l2"],
        help="Loss function type"
    )

    # Gradient estimation
    parser.add_argument(
        "--gradient-method",
        type=str,
        default="finite-diff",
        choices=["finite-diff", "auto"],
        help="Gradient estimation method (finite-diff recommended)"
    )
    parser.add_argument(
        "--fd-epsilon",
        type=float,
        default=0.01,
        help="Epsilon for finite difference gradient estimation"
    )
    parser.add_argument(
        "--no-parallel-fd",
        action="store_true",
        help="Disable parallel finite differences (slower but uses less memory)"
    )
    parser.add_argument(
        "--fd-batch-size",
        type=int,
        default=8,
        help="Batch size for parallel finite differences (higher = faster but more memory)"
    )

    # Text generation
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the voice optimization system.",
        help="Text to synthesize during optimization"
    )
    parser.add_argument(
        "--lang-code",
        type=str,
        default="a",
        help="Language code for Kokoro (a=American English, b=British, etc.)"
    )

    # Checkpointing
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save checkpoint every N iterations"
    )

    # Model settings
    parser.add_argument(
        "--wavlm-model",
        type=str,
        default="microsoft/wavlm-large",
        help="WavLM model to use for embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="hexgrad/Kokoro-82M",
        help="HuggingFace repo ID for Kokoro model"
    )

    args = parser.parse_args()

    # Validate inputs
    target_audio = Path(args.target_audio)
    if not target_audio.exists():
        print(f"Error: Target audio file not found: {target_audio}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("="*80)
    print("VOICE EMBEDDING GRADIENT DESCENT OPTIMIZER")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Target audio: {target_audio}")
    print(f"  Output directory: {output_dir}")
    print(f"  Text: {args.text}")
    print(f"  Device: {args.device}")
    print(f"  Language: {args.lang_code}")
    print(f"\nOptimization settings:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Gradient method: {args.gradient_method}")
    if args.gradient_method == "finite-diff":
        print(f"  FD epsilon: {args.fd_epsilon}")
        print(f"  FD parallelization: {not args.no_parallel_fd}")
        if not args.no_parallel_fd:
            print(f"  FD batch size: {args.fd_batch_size}")
    print(f"\nDimensionality reduction:")
    print(f"  Variable dimensions: {args.n_variable_dims}")
    print(f"  PCA components: {args.pca_components}")
    print(f"\nModels:")
    print(f"  WavLM: {args.wavlm_model}")
    print(f"  Kokoro repo: {args.repo_id}")
    print("="*80 + "\n")

    # Create configuration
    config = OptimizerConfig(
        n_variable_dims=args.n_variable_dims,
        pca_components=args.pca_components,
        learning_rate=args.learning_rate,
        num_iterations=args.iterations,
        optimizer_type=args.optimizer,
        loss_type=args.loss_type,
        save_every=args.save_every,
        device=args.device,
        wavlm_model=args.wavlm_model,
        text_for_generation=args.text
    )

    # Initialize Kokoro pipeline
    print("Initializing Kokoro TTS pipeline...")
    try:
        pipeline = KPipeline(
            lang_code=args.lang_code,
            repo_id=args.repo_id,
            device=args.device
        )
        print(f"Kokoro pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing Kokoro pipeline: {e}")
        sys.exit(1)

    # Create and setup optimizer
    try:
        optimizer = create_optimizer(
            voices_dir=args.voices_dir if Path(args.voices_dir).exists() else None,
            kokoro_pipeline=pipeline,
            config=config
        )
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run optimization
    try:
        results = optimizer.optimize(
            target_audio_path=target_audio,
            output_dir=output_dir,
            text=args.text,
            use_finite_diff=(args.gradient_method == "finite-diff"),
            fd_epsilon=args.fd_epsilon,
            parallel_fd=(not args.no_parallel_fd),
            fd_batch_size=args.fd_batch_size
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Final loss: {results['history']['loss'][-1]:.6f}")
    print(f"Total iterations: {len(results['history']['loss'])}")
    print(f"Output directory: {results['output_dir']}")
    print(f"\nOptimized voice files:")
    print(f"  Best: {results['output_dir'] / 'voice_best.pt'}")
    print(f"  Final: {results['output_dir'] / 'voice_final.pt'}")
    print(f"\nHistory: {results['output_dir'] / 'optimization_history.json'}")
    print("="*80)

    print("\nTo use the optimized voice with Kokoro:")
    print(f"  from kokoro import KPipeline")
    print(f"  import torch")
    print(f"  pipeline = KPipeline(lang_code='{args.lang_code}')")
    print(f"  voice = torch.load('{results['output_dir'] / 'voice_best.pt'}')")
    print(f"  for result in pipeline('Your text here', voice=voice):")
    print(f"      # Use result.audio")
    print()


if __name__ == "__main__":
    main()
