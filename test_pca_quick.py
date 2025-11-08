#!/usr/bin/env python3
"""Quick test of the PCA system with a small subset of voices."""

from kokoro.voice_pca_system import VoicePCASystem
from kokoro.synthetic_voices import SyntheticVoiceGenerator
import numpy as np

print("Testing PCA System...")
print()

# Test with a small subset of voices
test_voices = ['af_bella', 'af_sarah', 'am_adam', 'bf_alice', 'bm_george']

print(f"1. Initializing with {len(test_voices)} voices...")
pca_system = VoicePCASystem(
    voices_dir="voices",
    variance_coverage=0.99  # Use 99% for faster testing
)

print("\n2. Loading voices...")
pca_system.load_all_voices(voice_list=test_voices)
print(f"   Loaded: {len(pca_system.voice_data)} voices")

print("\n3. Calculating centroid...")
centroid = pca_system.calculate_centroid()
print(f"   Centroid shape: {centroid.shape}")

print("\n4. Fitting PCA...")
pca_system.fit_pca()
print(f"   Components: {pca_system.n_components}")
print(f"   Variance: {pca_system.explained_variance_ratio.sum() * 100:.2f}%")

print("\n5. Calculating bounds...")
pca_system.calculate_pca_bounds(max_exaggeration=5.0)
print(f"   Bounds calculated")

print("\n6. Testing voice transformation...")
voice = pca_system.get_voice_by_name('af_bella')
pca_coords = pca_system.voice_to_pca(voice)
reconstructed = pca_system.inverse_pca_transform(pca_coords)
print(f"   Original shape: {voice.shape}")
print(f"   PCA coords: {pca_coords.shape}")
print(f"   Reconstructed shape: {reconstructed.shape}")

print("\n7. Testing synthetic voice generation...")
generator = SyntheticVoiceGenerator(
    pca_system=pca_system,
    max_exaggeration=2.0,
    flatten_factor=0.3
)
synthetic_voice = generator.generate_random_voice()
print(f"   Generated: {synthetic_voice['name']}")
print(f"   Shape: {synthetic_voice['voice_tensor'].shape}")

print("\n✓ All tests passed!")
