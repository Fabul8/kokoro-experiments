/**
 * Examples demonstrating custom voice loading in Kokoro TTS
 *
 * This file shows how to use:
 * 1. Binary voices from custom URLs
 * 2. PCA-based voice reconstruction
 */

import { KokoroTTS } from '../src/kokoro.js';

async function main() {
  // Load the Kokoro TTS model
  const tts = await KokoroTTS.from_pretrained('onnx-community/Kokoro-82M-v1.0-ONNX', {
    dtype: 'fp32',
    device: 'cpu'
  });

  const text = "Hello, this is a test of custom voice loading.";

  // ===== Example 1: Using a preset voice (existing functionality) =====
  console.log("Example 1: Preset voice");
  const audio1 = await tts.generate(text, {
    voice: "af_heart"
  });
  console.log("Generated audio with preset voice af_heart");

  // ===== Example 2: Binary voice from custom URL =====
  console.log("\nExample 2: Binary voice from URL");
  const binaryVoice = {
    type: 'binary',
    url: 'https://example.com/my-custom-voice.bin',
    language: 'a' // Optional: 'a' for en-us, 'b' for en-gb
  };

  const audio2 = await tts.generate(text, {
    voice: binaryVoice
  });
  console.log("Generated audio with custom binary voice");

  // ===== Example 3: PCA-based voice reconstruction =====
  console.log("\nExample 3: PCA-based voice");

  // Define PCA weights (pre-computed on server)
  // These would typically come from a voice customization UI with sliders
  const pcaWeights = new Array(50).fill(0); // 50 principal components
  pcaWeights[0] = 0.5;   // Adjust first PC
  pcaWeights[1] = -0.2;  // Adjust second PC
  pcaWeights[2] = 0.8;   // Adjust third PC
  // ... etc

  const pcaVoice = {
    type: 'pca',
    centroidUrl: 'https://example.com/voices/centroid.bin',
    componentsUrl: 'https://example.com/voices/pca_components.bin',
    pcaValues: pcaWeights,
    language: 'a' // Optional
  };

  const audio3 = await tts.generate(text, {
    voice: pcaVoice
  });
  console.log("Generated audio with PCA-reconstructed voice");

  // ===== Example 4: Streaming with custom voices =====
  console.log("\nExample 4: Streaming with custom voice");
  const longText = "This is a longer text. It will be split into sentences. Each sentence will be synthesized separately.";

  for await (const { text: chunk, audio } of tts.stream(longText, {
    voice: binaryVoice
  })) {
    console.log(`Generated chunk: "${chunk}"`);
    // In a real application, you would play the audio chunks as they arrive
  }
}

// Only run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { main };
