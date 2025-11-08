# Kokoro Voice Manipulation Examples

This directory contains examples demonstrating the complete voice manipulation system for Kokoro TTS, including PCA-based voice creation, synthetic voice generation, and interactive GUI.

## Overview

The voice manipulation system provides:

1. **Centroid Calculation**: Calculate the mean voice across all loaded voices
2. **PCA Decomposition**: Decompose voices into principal components (99.9% variance coverage)
3. **Inverse PCA**: Reconstruct voices from PCA coordinates
4. **Exaggeration Support**: Create variations up to 5x the natural range (GUI) or 2x (synthetic)
5. **Synthetic Voice Generation**: Generate random voices with distribution-aware sampling
6. **Interactive GUI**: Slider-based interface for real-time voice creation

## Quick Start

### 1. Initialize the PCA System

First, load all voices and create the PCA system:

```bash
python examples/initialize_pca_system.py
```

This will:
- Load all available voices
- Calculate the centroid (mean voice)
- Fit PCA with 99.9% variance coverage
- Calculate bounds for 5x exaggeration
- Save the system to `voices/pca_system/`

**Output**: PCA system with typically 20-40 components covering 99.9% of variance

### 2. Generate Synthetic Voices

Generate random synthetic voices with distribution-aware sampling:

```bash
python examples/generate_synthetic_voices.py --n-voices 10
```

Options:
- `--n-voices N`: Number of voices to generate (default: 10)
- `--exaggeration X`: Max exaggeration factor (default: 2.0)
- `--flatten F`: Distribution flattening 0-1 (default: 0.3)
- `--distribution D`: Sampling distribution (normal/uniform/truncated_normal)
- `--tail-reduction T`: Tail reduction factor 0-1 (default: 0.7)
- `--output-dir DIR`: Output directory (default: voices/synthetic)
- `--name-style S`: Name style (random/fantasy/tech/nature)

**Output**: Synthetic voices saved to `voices/synthetic/`

### 3. Launch the Interactive GUI

Start the web-based GUI for creating custom voices:

```bash
python examples/launch_voice_gui.py
```

Options:
- `--port PORT`: Server port (default: 7860)
- `--share`: Create a public share link
- `--pca-system PATH`: Custom PCA system path

**Access**: Open http://localhost:7860 in your browser

### 4. Run the Complete Demo

See all features in action:

```bash
python examples/voice_manipulation_demo.py
```

This demonstrates:
- PCA system initialization
- Voice ↔ PCA transformation
- Custom voice creation
- Synthetic voice generation
- Saving and loading
- Statistical analysis

## System Architecture

### Components

```
kokoro/
├── voice_pca_system.py      # Core PCA system
├── synthetic_voices.py       # Random voice generation
└── voice_gui.py              # Interactive GUI

examples/
├── initialize_pca_system.py  # Setup script
├── generate_synthetic_voices.py  # Batch generation
├── launch_voice_gui.py       # GUI launcher
└── voice_manipulation_demo.py    # Complete demo
```

### Voice Data Flow

```
Original Voices
    ↓
[Load & Interpolate]
    ↓
Centroid Calculation
    ↓
[Center around centroid]
    ↓
PCA Fitting (99.9% variance)
    ↓
PCA Space (20-40 components)
    ↓
[Manipulate coordinates]
    ↓
Inverse PCA Transform
    ↓
Custom Voice
```

## Usage Examples

### Python API

#### Create PCA System

```python
from kokoro.voice_pca_system import VoicePCASystem

# Initialize
pca_system = VoicePCASystem(
    voices_dir="voices",
    variance_coverage=0.999  # 99.9%
)

# Load and fit
pca_system.load_all_voices()
pca_system.calculate_centroid()
pca_system.fit_pca()
pca_system.calculate_pca_bounds(max_exaggeration=5.0)

# Save for later
pca_system.save_system("voices/pca_system")
```

#### Transform Voice to PCA

```python
# Load a voice
voice = pca_system.get_voice_by_name("af_bella")

# Transform to PCA space
pca_coords = pca_system.voice_to_pca(voice)
print(f"PCA coordinates: {pca_coords[:5]}")

# Transform back
reconstructed = pca_system.inverse_pca_transform(pca_coords)
```

#### Create Custom Voice

```python
import numpy as np

# Create custom PCA coordinates
pca_coords = np.zeros(pca_system.n_components)
pca_coords[0] = 2.0   # Boost PC1
pca_coords[1] = -1.5  # Reduce PC2

# Generate voice
custom_voice = pca_system.inverse_pca_transform(pca_coords)

# Save
import torch
torch.save(custom_voice, "voices/custom/my_voice.pt")
```

#### Blend Two Voices

```python
# Get two voices
voice1 = pca_system.get_voice_by_name("af_bella")
voice2 = pca_system.get_voice_by_name("bf_alice")

# Transform to PCA
pca1 = pca_system.voice_to_pca(voice1)
pca2 = pca_system.voice_to_pca(voice2)

# Blend (70% + 30%)
blended_pca = 0.7 * pca1 + 0.3 * pca2
blended_voice = pca_system.inverse_pca_transform(blended_pca)
```

#### Generate Synthetic Voices

```python
from kokoro.synthetic_voices import SyntheticVoiceGenerator

# Initialize generator
generator = SyntheticVoiceGenerator(
    pca_system=pca_system,
    max_exaggeration=2.0,
    flatten_factor=0.3
)

# Generate single voice
voice = generator.generate_random_voice()
print(f"Generated: {voice['name']}")

# Generate batch
voices = generator.generate_batch(
    n_voices=10,
    save_dir="voices/synthetic"
)
```

## Technical Details

### PCA Configuration

- **Variance Coverage**: 99.9% (typically 20-40 components)
- **Input Dimension**: seq_len × 256 (flattened to single vector)
- **Centering**: All voices centered around global centroid
- **Interpolation**: Voices interpolated to median sequence length

### Exaggeration System

- **GUI Mode**: Up to 5x exaggeration for creative exploration
- **Synthetic Mode**: Up to 2x exaggeration for realistic variations
- **Bounds**: Calculated from actual voice data range

### Distribution-Aware Sampling

The synthetic voice generator uses sophisticated sampling:

1. **Flattening** (default 0.3): Reduces high-variance components, boosts low-variance
   - More expressivity across all dimensions
   - Prevents over-reliance on top components

2. **Tail Reduction** (default 0.7): Exponentially reduces extreme values
   - Smaller probability of going beyond data bounds
   - More natural-sounding synthetic voices

3. **Component Weighting**: Inverse variance weighting
   - Balanced sampling across components
   - Prevents dominance by high-variance components

### Voice Tensor Format

Voices are stored as PyTorch tensors with shape `[seq_len, 256]`:

- **Dimensions 0-127**: Decoder style (timbre/voice quality)
- **Dimensions 128-255**: Prosody style (rhythm/pitch/timing)

## GUI Features

The interactive GUI (`launch_voice_gui.py`) provides:

### Controls

- **50 Sliders**: One per PCA component (first 50 displayed)
- **Voice Name**: Custom name for saving
- **Presets**: Reset, Random, Load existing
- **Test Text**: Custom text for audio preview

### Actions

- **Generate Preview**: Synthesize audio with current settings
- **Save Voice**: Save to `voices/custom/`
- **Load Voice**: Load and edit existing voices
- **Reset**: Return to centroid (all zeros)
- **Random**: Generate random PCA coordinates

### Display

- **Voice Statistics**: Real-time stats (distance, magnitude, etc.)
- **Audio Player**: Listen to generated voice
- **Component Info**: Variance explained by each component

## Troubleshooting

### PCA system not found

```
ERROR: PCA system not found!
```

**Solution**: Run `python examples/initialize_pca_system.py` first

### Import errors

```
ModuleNotFoundError: No module named 'kokoro'
```

**Solution**: Run from the `examples/` directory or add parent to path

### CUDA out of memory

**Solution**: The system defaults to CPU. For large voice sets, consider:
- Reducing `variance_coverage` (e.g., 0.99 instead of 0.999)
- Processing voices in batches

### GUI not accessible

**Solution**: Check firewall settings, try `--port 8080` or `--share` for public link

## Performance

### Initialization

- Load 50 voices: ~10-15 seconds
- Calculate centroid: ~1 second
- Fit PCA: ~2-5 seconds
- Total: ~15-20 seconds

### Generation

- Single voice from PCA: <0.1 seconds
- Batch of 100 voices: ~5 seconds
- GUI slider update: Real-time (<0.05 seconds)

### Memory

- PCA system: ~50-100 MB
- 50 voices loaded: ~30 MB
- Per-voice: ~0.5 MB

## Future Enhancements

Potential additions:

1. **Voice interpolation paths**: Smooth transitions between voices
2. **Semantic controls**: High-level controls (age, pitch, energy)
3. **Voice clustering**: Automatic grouping by characteristics
4. **Voice morphing**: Temporal interpolation for dynamic effects
5. **Optimization**: Pre-compute common operations

## References

- **Kokoro TTS**: https://github.com/hexgrad/Kokoro-82M
- **PCA**: Principal Component Analysis for dimensionality reduction
- **Gradio**: https://gradio.app/ (GUI framework)

## License

Same as Kokoro TTS parent project.
