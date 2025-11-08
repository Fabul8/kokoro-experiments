# Kokoro Voice Analysis Tools

This repository contains comprehensive analysis tools for the Kokoro-82M text-to-speech model, focusing on voice embedding analysis, architecture exploration, and visualization.

## Overview

Kokoro-82M is an 82-million parameter TTS model with a unique architecture based on StyleTTS2. This analysis suite provides tools to:

1. **Download and manage all 63 voice embeddings**
2. **Analyze the model architecture** (layers, parameters, data flow)
3. **Perform PCA analysis** on voice embeddings
4. **Calculate centroids** for voice groups
5. **Analyze inter-layer correlations** between decoder and prosody layers
6. **Generate comprehensive visualizations**

## Architecture Summary

### Model Components

**Kokoro-82M** consists of ~82M parameters distributed across:

1. **BERT Encoder (CustomAlbert)**: Contextual phoneme embeddings
   - Vocabulary size: configurable
   - Hidden size: 768 (typical)
   - Transformer layers with attention

2. **Prosody Predictor**: Predicts duration, F0 (pitch), and N (noise)
   - Duration encoder with adaptive layer normalization
   - LSTM-based prediction
   - F0 and N branches with AdaIN residual blocks

3. **Text Encoder**: CNN + LSTM architecture
   - Embedding layer for phonemes
   - Multiple CNN layers with dropout
   - Bidirectional LSTM

4. **Decoder (iSTFTNet)**: Neural vocoder
   - AdaIN-based encoding/decoding
   - Source-filter inspired generator
   - STFT/iSTFT operations for audio synthesis

### Voice Embeddings

Each voice embedding is a tensor of shape `[seq_len, 256]`:
- **Dimensions 0-127**: Style embedding for the decoder (controls timbre)
- **Dimensions 128-255**: Style embedding for the prosody predictor (controls rhythm/pitch)

The embedding is indexed by phoneme count, allowing different prosody for different sequence positions.

## Quick Start

### Setup

```bash
# Run the complete setup script
./scripts/setup_kokoro.sh
```

This will:
- Install all dependencies (including espeak-ng)
- Download all 63 voice embeddings
- Download model weights
- Run architecture and voice analysis
- Generate all visualizations

### Manual Installation

```bash
# Install dependencies
pip install -e .
pip install scikit-learn matplotlib seaborn pandas huggingface_hub soundfile tqdm

# Install espeak-ng (Linux)
sudo apt-get install espeak-ng

# Install espeak-ng (macOS)
brew install espeak-ng
```

## Usage

### Download All Voices

```bash
python scripts/download_all_voices.py
```

Downloads all 63 voices from HuggingFace to `voices/` directory.

### Analyze Architecture

```bash
python scripts/analyze_architecture.py
```

Generates:
- `voices/analysis/architecture_analysis.json` - Detailed JSON analysis
- `voices/analysis/architecture_report.txt` - Human-readable report
- Module-wise parameter counts
- Layer structure analysis
- Data flow documentation

### Analyze Voice Embeddings

```bash
python scripts/analyze_voices.py
```

Generates comprehensive analysis including:

**Analysis Results:**
- `voices/analysis/centroids.npz` - Centroid embeddings for all groups
- `voices/analysis/centroid_stats.json` - Centroid statistics
- `voices/analysis/analysis_report.txt` - Summary report
- `voices/analysis/top_variable_dimensions.json` - Most variable dimensions

**Visualizations:**
- `pca_variance.png` - Explained variance for each PCA component
- `pca_2d.png` - 2D PCA projections colored by voice groups
- `tsne_both.png` - t-SNE visualization of voice embeddings
- `interlayer_correlation.png` - Correlation heatmap between decoder and prosody layers
- `embedding_distributions.png` - Distribution of embedding values
- `dimension_variance.png` - Variance across voices per dimension

### Test Kokoro

```bash
python test_kokoro.py
```

Quick sanity check to ensure everything is working.

## Analysis Details

### PCA Analysis

Principal Component Analysis is performed on three embedding types:
1. **Decoder layer** (dimensions 0-127)
2. **Prosody layer** (dimensions 128-255)
3. **Both layers** (all 256 dimensions)

**Key Questions Answered:**
- How much variance is explained by the first N components?
- Which dimensions are most important for voice identity?
- How separable are different voice groups in PCA space?

### Centroid Analysis

Centroids are calculated for:
- **Overall**: Mean of all voice embeddings
- **Voice groups**: Mean per group (af, am, bf, bm, ef, etc.)
  - `af` = American Female
  - `am` = American Male
  - `bf` = British Female
  - `bm` = British Male
  - etc.

**Metrics:**
- Distance from overall centroid (measures group distinctiveness)
- Within-group variance
- Between-group distances

### Inter-Layer Correlation

Analyzes the correlation between:
- Decoder layer dimensions (0-127)
- Prosody layer dimensions (128-255)

**Key Insights:**
- Which decoder dimensions correlate with prosody?
- Are the layers independent or coupled?
- Which dimension pairs are most correlated?

### Dimension Variance Analysis

Identifies the most variable dimensions across all voices:
- High variance dimensions capture important voice characteristics
- Low variance dimensions may be redundant or less important

## Voice Groups

### American English
- **Female (11)**: af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
- **Male (9)**: am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa

### British English
- **Female (4)**: bf_alice, bf_emma, bf_isabella, bf_lily
- **Male (4)**: bm_daniel, bm_fable, bm_george, bm_lewis

### Other Languages
- **Spanish**: ef_dora, em_alex, em_santa
- **French**: ff_siwis
- **Hindi**: hf_alpha, hf_beta, hm_omega, hm_psi
- **Italian**: if_sara, im_nicola
- **Japanese**: jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
- **Portuguese**: pf_dora, pm_alex, pm_santa
- **Chinese**: zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi

## Expected Results

### Architecture Analysis

```
Total Parameters: ~82,000,000
Module Breakdown:
  - BERT Encoder: ~40-50% of parameters
  - Decoder: ~30-40% of parameters
  - Prosody Predictor: ~10-15% of parameters
  - Text Encoder: ~5-10% of parameters
```

### Voice Analysis

**PCA Results:**
- First principal component typically explains 15-30% of variance
- First 10 components typically explain 70-85% of variance
- Clear clustering by language/accent in PCA space

**Inter-Layer Correlation:**
- Generally low to moderate correlation between layers
- Some dimension pairs show high correlation (>0.7)
- Indicates partial independence of decoder and prosody representations

**Dimension Variance:**
- Top 10-20 dimensions capture most voice variation
- Many dimensions have low variance (may be less critical)

## Directory Structure

```
kokoro-experiments/
├── voices/                    # Voice embeddings
│   ├── af_heart.pt
│   ├── am_adam.pt
│   └── ... (63 total)
│   └── analysis/             # Analysis outputs
│       ├── architecture_analysis.json
│       ├── architecture_report.txt
│       ├── analysis_report.txt
│       ├── centroids.npz
│       ├── centroid_stats.json
│       ├── top_variable_dimensions.json
│       ├── pca_variance.png
│       ├── pca_2d.png
│       ├── tsne_both.png
│       ├── interlayer_correlation.png
│       ├── embedding_distributions.png
│       └── dimension_variance.png
├── scripts/
│   ├── download_all_voices.py
│   ├── analyze_voices.py
│   ├── analyze_architecture.py
│   └── setup_kokoro.sh
├── kokoro/                   # Main package
│   ├── model.py
│   ├── pipeline.py
│   ├── modules.py
│   └── istftnet.py
└── outputs/
    ├── audio/               # Generated audio files
    └── graphs/              # Additional visualizations
```

## Advanced Usage

### Custom Analysis

```python
from scripts.analyze_voices import VoiceAnalyzer

# Initialize analyzer
analyzer = VoiceAnalyzer('voices')

# Load voices
analyzer.load_voices()

# Run specific analyses
centroids = analyzer.compute_centroids()
pca_results = analyzer.pca_analysis(n_components=50)
corr_results = analyzer.inter_layer_correlation()

# Generate specific plots
analyzer.plot_pca_2d(pca_results)
analyzer.plot_tsne(layer='decoder', perplexity=30)
```

### Architecture Exploration

```python
from scripts.analyze_architecture import ArchitectureAnalyzer

analyzer = ArchitectureAnalyzer()
analyzer.load_model()

# Analyze specific modules
bert_analysis = analyzer.analyze_bert_encoder()
decoder_analysis = analyzer.analyze_decoder()
flow_analysis = analyzer.analyze_data_flow()
```

## Key Findings

### Voice Embedding Structure

1. **Dual-layer design**: Separate embeddings for decoder (timbre) and prosody (rhythm/pitch)
2. **Variable length**: Indexed by phoneme count for position-dependent control
3. **High dimensionality**: 256 total dimensions provide rich voice representation

### Most Relevant Layers/Dimensions

Based on variance analysis:
- **Decoder layer**: Dimensions with highest variance capture timbre differences
- **Prosody layer**: Dimensions with highest variance capture speaking style
- **Top 10-20 dimensions** per layer capture majority of voice variation

### Inter-Layer Correlates

- Moderate correlation between some decoder and prosody dimensions
- Suggests partial coupling between timbre and prosody
- Some dimensions are highly specialized (low correlation)

## Troubleshooting

### espeak-ng not found
```bash
# Linux
sudo apt-get install espeak-ng

# macOS
brew install espeak-ng
```

### CUDA out of memory
```python
# Use CPU instead
pipeline = KPipeline(lang_code='a', device='cpu')
```

### Missing voices
```bash
# Re-download all voices
python scripts/download_all_voices.py
```

## Contributing

To add new analysis:

1. Create analysis script in `scripts/`
2. Add visualization to `VoiceAnalyzer` class
3. Update `setup_kokoro.sh` to run analysis
4. Document in this README

## References

- [Kokoro-82M Model](https://huggingface.co/hexgrad/Kokoro-82M)
- [StyleTTS2](https://github.com/yl4579/StyleTTS2)
- [Kokoro Package](https://pypi.org/project/kokoro/)

## License

Apache 2.0 (same as Kokoro-82M)
