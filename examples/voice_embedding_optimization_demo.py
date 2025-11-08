#!/usr/bin/env python3
"""
Voice Embedding Optimization Demo

This demo shows how to use the voice embedding optimizer to match a target voice.

Steps:
1. Generate a reference audio with a known voice
2. Optimize a new voice embedding to match that audio
3. Compare the results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro import KPipeline
from kokoro.voice_embedding_optimizer import (
    OptimizerConfig,
    create_optimizer
)
import torch
import torchaudio


def generate_reference_audio():
    """Generate reference audio with a known voice."""
    print("Generating reference audio...")

    # Initialize pipeline
    pipeline = KPipeline(
        lang_code='a',  # American English
        repo_id='hexgrad/Kokoro-82M',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Generate audio with af_bella voice
    text = "Hello, this is a test of the voice optimization system."
    voice = 'af_bella'

    audio_chunks = []
    for result in pipeline(text, voice=voice):
        if result.audio is not None:
            audio_chunks.append(result.audio)

    audio = torch.cat(audio_chunks, dim=-1)

    # Save reference audio
    output_path = Path("outputs/reference_audio.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(
        str(output_path),
        audio.unsqueeze(0).cpu(),
        sample_rate=24000
    )

    print(f"Reference audio saved to: {output_path}")
    return output_path, text


def optimize_voice(reference_audio_path, text):
    """Optimize voice embedding to match reference audio."""
    print("\nOptimizing voice embedding...")

    # Configuration with quick settings for demo
    config = OptimizerConfig(
        n_variable_dims=128,  # Fewer dims for faster demo
        pca_components=64,    # Fewer components for faster demo
        learning_rate=0.05,
        num_iterations=100,   # Fewer iterations for demo
        optimizer_type="adam",
        loss_type="cosine",
        save_every=25,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        wavlm_model="microsoft/wavlm-large",
        text_for_generation=text
    )

    # Initialize Kokoro pipeline
    pipeline = KPipeline(
        lang_code='a',
        repo_id='hexgrad/Kokoro-82M',
        device=config.device
    )

    # Create optimizer
    optimizer = create_optimizer(
        voices_dir="voices" if Path("voices").exists() else None,
        kokoro_pipeline=pipeline,
        config=config
    )

    # Run optimization
    results = optimizer.optimize(
        target_audio_path=reference_audio_path,
        output_dir="outputs/voice_optimization_demo",
        text=text,
        use_finite_diff=True,
        fd_epsilon=0.01
    )

    return results


def test_optimized_voice(voice_path, text):
    """Test the optimized voice."""
    print("\nTesting optimized voice...")

    # Initialize pipeline
    pipeline = KPipeline(
        lang_code='a',
        repo_id='hexgrad/Kokoro-82M',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load optimized voice
    optimized_voice = torch.load(voice_path, weights_only=True)

    # Generate audio
    audio_chunks = []
    for result in pipeline(text, voice=optimized_voice):
        if result.audio is not None:
            audio_chunks.append(result.audio)

    audio = torch.cat(audio_chunks, dim=-1)

    # Save
    output_path = Path("outputs/optimized_audio.wav")
    torchaudio.save(
        str(output_path),
        audio.unsqueeze(0).cpu(),
        sample_rate=24000
    )

    print(f"Optimized audio saved to: {output_path}")


def main():
    """Run the demo."""
    print("="*80)
    print("VOICE EMBEDDING OPTIMIZATION DEMO")
    print("="*80)
    print("\nThis demo will:")
    print("1. Generate reference audio with a known voice")
    print("2. Optimize a new voice embedding to match that audio")
    print("3. Generate audio with the optimized voice")
    print("="*80 + "\n")

    # Step 1: Generate reference
    reference_path, text = generate_reference_audio()

    # Step 2: Optimize
    results = optimize_voice(reference_path, text)

    # Step 3: Test
    best_voice_path = results['output_dir'] / "voice_best.pt"
    test_optimized_voice(best_voice_path, text)

    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  Reference audio: outputs/reference_audio.wav")
    print(f"  Optimized voice: {best_voice_path}")
    print(f"  Optimized audio: outputs/optimized_audio.wav")
    print(f"  Optimization history: {results['output_dir'] / 'optimization_history.json'}")
    print("\nCompare the reference and optimized audio to see how well it matched!")
    print("="*80)


if __name__ == "__main__":
    main()
