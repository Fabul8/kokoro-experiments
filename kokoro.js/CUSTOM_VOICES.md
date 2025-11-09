# Custom Voice Loading in Kokoro TTS

This document describes the custom voice loading features added to Kokoro TTS, which allow you to use:
1. **Binary voices** - Custom voice files loaded from URLs
2. **PCA-based voices** - Voices reconstructed from PCA components with customizable weights

## Architecture Overview

### Voice Data Format

Kokoro TTS uses voice embeddings of shape `[510, 256]` (130,560 floats total):
- **510 frames** corresponding to different text lengths
- **256 dimensions** for each style vector

The model selects the appropriate frame based on the number of input tokens, then uses the 256-dimensional style vector for synthesis.

### Voice Types

#### 1. Preset Voices (Existing)
```javascript
const audio = await tts.generate(text, {
  voice: "af_heart"
});
```

#### 2. Binary Voices
Load a custom voice from a `.bin` file URL:
```javascript
const audio = await tts.generate(text, {
  voice: {
    type: 'binary',
    url: 'https://example.com/my-voice.bin',
    language: 'a' // Optional: 'a' for en-us, 'b' for en-gb
  }
});
```

**Binary Format:**
- Raw Float32 array
- 130,560 floats (510 × 256)
- Little-endian byte order

#### 3. PCA-based Voices
Reconstruct a voice from PCA components:
```javascript
const audio = await tts.generate(text, {
  voice: {
    type: 'pca',
    centroidUrl: 'https://example.com/centroid.bin',
    componentsUrl: 'https://example.com/pca_components.bin',
    pcaValues: [0.5, -0.2, 0.8, ...], // Array of PCA weights
    language: 'a' // Optional
  }
});
```

## PCA Voice Reconstruction

### Mathematical Formula

Given:
- **centroid**: ℝ^130560 (mean voice embedding)
- **PCA_components**: ℝ^(K×130560) (K principal components)
- **pcaValues**: ℝ^K (PCA weights)

The reconstructed voice is computed as:

```
voice = centroid + Σ(i=0 to K-1) pcaValues[i] × PCA_components[i]
```

Or in matrix notation:
```
voice = centroid + PCA_components^T × pcaValues
```

### Implementation Details

The reconstruction happens in `voices.js:reconstructVoiceFromPCA()`:

```javascript
function reconstructVoiceFromPCA(centroid, components, pcaValues) {
  const N = 130560;  // 510 * 256
  const K = pcaValues.length;

  // voice = centroid (clone)
  const voice = new Float32Array(centroid);

  // voice += Σ pcaValues[i] × PCA_components[i]
  for (let i = 0; i < K; i++) {
    const weight = pcaValues[i];
    const pcOffset = i * N;
    for (let j = 0; j < N; j++) {
      voice[j] += weight * components[pcOffset + j];
    }
  }

  return voice;
}
```

### File Formats

#### Centroid File (`centroid.bin`)
- **Format**: Raw Float32 array
- **Shape**: [130560] (flattened from [510, 256])
- **Size**: 522,240 bytes (130,560 × 4 bytes)

#### PCA Components File (`pca_components.bin`)
- **Format**: Raw Float32 array
- **Shape**: [K × 130560] where K is the number of PCs
- **Layout**: Flattened, with PC0 first, then PC1, etc.
- **Size**: K × 522,240 bytes

For example, with 50 PCs:
- Size: 50 × 522,240 = 26,112,000 bytes (~26 MB)

#### PCA Values
- **Format**: Array of floats
- **Length**: K (must match number of PCs in components file)
- **Computed on server** from slider values or other UI inputs

## Caching

All external resources are cached using the browser's Cache API:
- **Cache name**: `"kokoro-voices"`
- Binary files are cached after first fetch
- PCA reconstructions are cached based on the combination of URLs and pcaValues

### Cache Keys
- **Binary voices**: URL
- **PCA voices**: JSON.stringify({centroidUrl, componentsUrl, pcaValues})

## Usage Examples

### Example 1: Simple Binary Voice
```javascript
const tts = await KokoroTTS.from_pretrained('onnx-community/Kokoro-82M-v1.0-ONNX');

const audio = await tts.generate("Hello, world!", {
  voice: {
    type: 'binary',
    url: 'https://cdn.example.com/voices/custom-voice.bin'
  }
});
```

### Example 2: PCA Voice with 50 Components
```javascript
// These weights would typically come from a voice customization UI
const pcaWeights = new Array(50).fill(0);
pcaWeights[0] = 0.5;   // Adjust pitch/tone
pcaWeights[1] = -0.3;  // Adjust speed variation
// ... etc

const audio = await tts.generate("Custom voice test", {
  voice: {
    type: 'pca',
    centroidUrl: 'https://cdn.example.com/voices/centroid.bin',
    componentsUrl: 'https://cdn.example.com/voices/pca_50.bin',
    pcaValues: pcaWeights
  }
});
```

### Example 3: Streaming with Custom Voice
```javascript
const voice = {
  type: 'pca',
  centroidUrl: 'https://cdn.example.com/voices/centroid.bin',
  componentsUrl: 'https://cdn.example.com/voices/pca_50.bin',
  pcaValues: Array(50).fill(0) // Neutral voice
};

for await (const { text, audio } of tts.stream(longText, { voice })) {
  // Play audio chunk
  await playAudio(audio);
}
```

## Implementation Files

### Modified Files

1. **`src/voices.js`**
   - Added `loadBinaryFromUrl()` for fetching binary files with caching
   - Added `reconstructVoiceFromPCA()` for PCA reconstruction
   - Extended `getVoiceData()` to handle voice objects
   - Added separate caches for binary and PCA voices

2. **`src/kokoro.js`**
   - Updated TypeScript definitions for voice types
   - Modified `_validate_voice()` to handle voice objects
   - Added support for optional `language` field in custom voices

### New Files

1. **`examples/custom-voices.js`**
   - Usage examples for all voice types
   - Demonstrates preset, binary, and PCA voices

2. **`CUSTOM_VOICES.md`**
   - This documentation file

## API Reference

### Voice Object Types

#### VoiceBinary
```typescript
{
  type: 'binary',
  url: string,
  language?: 'a' | 'b'  // 'a' = en-us, 'b' = en-gb
}
```

#### VoicePCA
```typescript
{
  type: 'pca',
  centroidUrl: string,
  componentsUrl: string,
  pcaValues: number[],
  language?: 'a' | 'b'
}
```

### Functions

#### `getVoiceData(voice)`
Loads voice data from various sources.

**Parameters:**
- `voice`: `string | VoiceBinary | VoicePCA`

**Returns:** `Promise<Float32Array>`

**Example:**
```javascript
// Preset voice
const voice1 = await getVoiceData("af_heart");

// Binary voice
const voice2 = await getVoiceData({
  type: 'binary',
  url: 'https://example.com/voice.bin'
});

// PCA voice
const voice3 = await getVoiceData({
  type: 'pca',
  centroidUrl: 'https://example.com/centroid.bin',
  componentsUrl: 'https://example.com/components.bin',
  pcaValues: [0, 0, 0, ...]
});
```

## Performance Considerations

### Caching Strategy
- URLs and reconstructed voices are cached to avoid re-downloading and re-computing
- PCA reconstruction is cached based on the full voice configuration
- Changing any `pcaValue` will trigger a new reconstruction

### Memory Usage
- Each voice (preset or custom) uses 522,240 bytes (~522 KB)
- PCA components file can be large (e.g., 50 PCs = ~26 MB)
- Centroid is the same size as a regular voice (~522 KB)

### Optimization Tips
1. **Reuse voice objects** when possible to benefit from caching
2. **Minimize PCA components** - use only as many as needed for good quality
3. **Host files on CDN** with appropriate cache headers
4. **Use compression** for serving binary files (gzip/brotli)

## Server-Side Integration

### Generating PCA Files

The server should provide:
1. **Centroid**: Computed from all voice samples
2. **PCA Components**: Top K eigenvectors from PCA decomposition
3. **PCA Values**: Computed from user inputs (sliders, presets, etc.)

Example Python code for PCA computation:
```python
import numpy as np
from sklearn.decomposition import PCA

# Load voice samples: shape [N_samples, 130560]
voices = load_voice_samples()

# Compute PCA
pca = PCA(n_components=50)
pca.fit(voices)

# Save centroid
centroid = np.mean(voices, axis=0)
centroid.astype(np.float32).tofile('centroid.bin')

# Save components
components = pca.components_  # Shape: [50, 130560]
components.astype(np.float32).tofile('pca_components.bin')

# For a specific voice, compute PCA values
voice = load_specific_voice()
pca_values = pca.transform((voice - centroid).reshape(1, -1))[0]
```

## Browser Compatibility

The custom voice loading features use:
- **Cache API**: Supported in all modern browsers
- **Fetch API**: Supported in all modern browsers
- **Float32Array**: Supported in all modern browsers

No additional polyfills are required for modern browsers.

## Troubleshooting

### Issue: Voice not loading
- Check that the URL is accessible (CORS headers)
- Verify the binary file format (Float32, little-endian)
- Ensure file size matches expected size (130,560 floats = 522,240 bytes)

### Issue: PCA reconstruction sounds wrong
- Verify centroid and components files are correct
- Check that pcaValues length matches number of components
- Ensure pcaValues are in the correct range (typically -5 to +5)

### Issue: High memory usage
- Reduce number of PCA components
- Clear voice caches periodically if needed
- Avoid creating many different voice configurations

## Future Enhancements

Possible future improvements:
1. Support for compressed voice formats
2. Incremental PCA updates
3. Voice interpolation between presets
4. Real-time PCA weight adjustment
5. Voice mixing/blending capabilities
