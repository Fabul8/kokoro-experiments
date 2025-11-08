#!/usr/bin/env python3
"""
Analyze inter-frame variation in Kokoro voice embeddings.

Voice embeddings have shape [510, 1, 256] where:
- 510 frames correspond to different phoneme sequence lengths
- Each frame is a 256-dimensional style vector

This script analyzes:
1. How much variation exists between consecutive frames
2. Whether frames are interpolated or independent
3. Which frames are most unique
4. Variance distribution across decoder vs prosody layers
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import json

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class FrameVariationAnalyzer:
    """Analyzer for inter-frame variation in voice embeddings."""

    def __init__(self, voices_dir: str = 'voices', output_dir: str = 'voices/analysis'):
        self.voices_dir = Path(voices_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voices = {}

    def load_voices(self):
        """Load all voice files."""
        voice_files = list(self.voices_dir.glob('*.pt'))

        print(f"Loading {len(voice_files)} voices...")
        for voice_file in voice_files:
            voice_name = voice_file.stem
            voice_tensor = torch.load(voice_file, map_location='cpu', weights_only=True)
            self.voices[voice_name] = voice_tensor.squeeze()  # [510, 256]

        print(f"Loaded {len(self.voices)} voices")
        return len(self.voices)

    def analyze_single_voice_frames(self, voice_name: str):
        """Analyze frame variation for a single voice."""
        voice = self.voices[voice_name]  # [510, 256]

        results = {
            'voice_name': voice_name,
            'num_frames': voice.shape[0],
            'frame_dim': voice.shape[1],
        }

        # Consecutive frame similarity
        consecutive_sim = []
        for i in range(voice.shape[0] - 1):
            sim = cosine_similarity(
                voice[i:i+1].numpy(),
                voice[i+1:i+2].numpy()
            )[0, 0]
            consecutive_sim.append(sim)

        results['consecutive_similarity'] = {
            'mean': float(np.mean(consecutive_sim)),
            'std': float(np.std(consecutive_sim)),
            'min': float(np.min(consecutive_sim)),
            'max': float(np.max(consecutive_sim)),
        }

        # Frame variance
        frame_variance = voice.var(dim=0).numpy()  # Variance across frames for each dimension

        results['dimension_variance'] = {
            'decoder_mean': float(frame_variance[:128].mean()),
            'prosody_mean': float(frame_variance[128:].mean()),
            'decoder_std': float(frame_variance[:128].std()),
            'prosody_std': float(frame_variance[128:].std()),
            'total_mean': float(frame_variance.mean()),
        }

        # PCA on frames
        pca = PCA(n_components=min(50, voice.shape[0]))
        pca.fit(voice.numpy())

        results['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()[:10],
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()[:10],
            'components_for_95': int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1),
            'components_for_99': int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.99) + 1),
        }

        return results

    def analyze_all_voices(self):
        """Analyze frame variation across all voices."""
        print("\n" + "="*80)
        print("ANALYZING FRAME VARIATION ACROSS ALL VOICES")
        print("="*80 + "\n")

        all_results = {}

        for voice_name in sorted(self.voices.keys())[:10]:  # Sample first 10 for speed
            print(f"Analyzing {voice_name}...")
            results = self.analyze_single_voice_frames(voice_name)
            all_results[voice_name] = results

            print(f"  Consecutive similarity: {results['consecutive_similarity']['mean']:.6f}")
            print(f"  Components for 95% variance: {results['pca']['components_for_95']}")
            print(f"  Components for 99% variance: {results['pca']['components_for_99']}")

        # Save results
        with open(self.output_dir / 'frame_variation_analysis.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        return all_results

    def analyze_frame_differences(self):
        """Analyze how frames differ across the sequence."""
        print("\n" + "="*80)
        print("FRAME DIFFERENCE ANALYSIS")
        print("="*80 + "\n")

        # Use first voice as representative
        voice_name = list(self.voices.keys())[0]
        voice = self.voices[voice_name]  # [510, 256]

        print(f"Analyzing frame differences for: {voice_name}")

        # Compute pairwise distances between all frames
        n_frames = voice.shape[0]
        sample_indices = np.linspace(0, n_frames-1, min(100, n_frames), dtype=int)

        sampled_frames = voice[sample_indices].numpy()

        # Cosine similarity matrix
        sim_matrix = cosine_similarity(sampled_frames)

        # Plot similarity matrix
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Similarity heatmap
        ax = axes[0]
        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0.9, vmax=1.0)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Frame Index')
        ax.set_title('Inter-Frame Cosine Similarity')
        plt.colorbar(im, ax=ax, label='Similarity')

        # Consecutive frame similarity
        ax = axes[1]
        consecutive_sim = []
        for i in range(n_frames - 1):
            sim = cosine_similarity(
                voice[i:i+1].numpy(),
                voice[i+1:i+2].numpy()
            )[0, 0]
            consecutive_sim.append(sim)

        ax.plot(consecutive_sim, alpha=0.7)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Similarity to Next Frame')
        ax.set_title('Consecutive Frame Similarity')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.999, color='r', linestyle='--', alpha=0.5, label='99.9% threshold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_similarity_analysis.png', bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'frame_similarity_analysis.png'}")
        plt.close()

    def analyze_decoder_vs_prosody_variation(self):
        """Compare variation between decoder and prosody layers across frames."""
        print("\n" + "="*80)
        print("DECODER VS PROSODY LAYER VARIATION")
        print("="*80 + "\n")

        decoder_variances = []
        prosody_variances = []

        for voice_name, voice in list(self.voices.items())[:10]:
            frame_var = voice.var(dim=0).numpy()
            decoder_variances.append(frame_var[:128])
            prosody_variances.append(frame_var[128:])

        decoder_var = np.array(decoder_variances).mean(axis=0)
        prosody_var = np.array(prosody_variances).mean(axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Decoder variance per dimension
        ax = axes[0, 0]
        ax.bar(range(128), decoder_var, alpha=0.7, color='blue')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Variance Across Frames')
        ax.set_title('Decoder Layer: Inter-Frame Variance per Dimension')
        ax.grid(True, alpha=0.3)

        # Prosody variance per dimension
        ax = axes[0, 1]
        ax.bar(range(128), prosody_var, alpha=0.7, color='orange')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Variance Across Frames')
        ax.set_title('Prosody Layer: Inter-Frame Variance per Dimension')
        ax.grid(True, alpha=0.3)

        # Comparison histogram
        ax = axes[1, 0]
        ax.hist(decoder_var, bins=50, alpha=0.5, label='Decoder', color='blue')
        ax.hist(prosody_var, bins=50, alpha=0.5, label='Prosody', color='orange')
        ax.set_xlabel('Variance')
        ax.set_ylabel('Count')
        ax.set_title('Variance Distribution Comparison')
        ax.legend()
        ax.set_yscale('log')

        # Box plot
        ax = axes[1, 1]
        ax.boxplot([decoder_var, prosody_var], labels=['Decoder', 'Prosody'])
        ax.set_ylabel('Inter-Frame Variance')
        ax.set_title('Variance Comparison (Box Plot)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_variance_comparison.png', bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'layer_variance_comparison.png'}")
        plt.close()

        print(f"\nDecoder layer variance: {decoder_var.mean():.6f} ± {decoder_var.std():.6f}")
        print(f"Prosody layer variance: {prosody_var.mean():.6f} ± {prosody_var.std():.6f}")
        print(f"Prosody/Decoder ratio: {prosody_var.mean() / decoder_var.mean():.2f}x")

    def analyze_unique_frames(self):
        """Identify which frames are most unique."""
        print("\n" + "="*80)
        print("UNIQUE FRAME ANALYSIS")
        print("="*80 + "\n")

        voice_name = list(self.voices.keys())[0]
        voice = self.voices[voice_name]  # [510, 256]

        # Compute distance from mean
        mean_frame = voice.mean(dim=0, keepdim=True)
        distances = torch.norm(voice - mean_frame, dim=1).numpy()

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        ax = axes[0]
        ax.plot(distances, linewidth=1)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Distance from Mean Frame')
        ax.set_title('Frame Uniqueness (Distance from Mean)')
        ax.grid(True, alpha=0.3)

        # Mark most unique frames
        top_k = 10
        top_indices = np.argsort(distances)[-top_k:]
        ax.scatter(top_indices, distances[top_indices], c='red', s=50, zorder=5,
                   label=f'Top {top_k} most unique')
        ax.legend()

        # Zoom into first 50 frames
        ax = axes[1]
        ax.plot(distances[:50], linewidth=2)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Distance from Mean Frame')
        ax.set_title('Frame Uniqueness (First 50 Frames)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'unique_frames.png', bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'unique_frames.png'}")
        plt.close()

        print(f"\nMost unique frames: {top_indices}")
        print(f"Frame 0 distance: {distances[0]:.6f}")
        print(f"Frame 50 distance: {distances[50]:.6f}")
        print(f"Frame 509 distance: {distances[509]:.6f}")

    def generate_report(self, all_results):
        """Generate comprehensive report."""
        report = []
        report.append("="*80)
        report.append("FRAME VARIATION ANALYSIS REPORT")
        report.append("="*80)
        report.append("")

        # Aggregate statistics
        avg_consecutive_sim = np.mean([r['consecutive_similarity']['mean'] for r in all_results.values()])
        avg_decoder_var = np.mean([r['dimension_variance']['decoder_mean'] for r in all_results.values()])
        avg_prosody_var = np.mean([r['dimension_variance']['prosody_mean'] for r in all_results.values()])
        avg_components_95 = np.mean([r['pca']['components_for_95'] for r in all_results.values()])
        avg_components_99 = np.mean([r['pca']['components_for_99'] for r in all_results.values()])

        report.append("SUMMARY STATISTICS (across sampled voices)")
        report.append("-"*80)
        report.append(f"Average consecutive frame similarity: {avg_consecutive_sim:.6f}")
        report.append(f"Average decoder layer variance: {avg_decoder_var:.6f}")
        report.append(f"Average prosody layer variance: {avg_prosody_var:.6f}")
        report.append(f"Prosody/Decoder variance ratio: {avg_prosody_var/avg_decoder_var:.2f}x")
        report.append(f"Average components for 95% variance: {avg_components_95:.1f}")
        report.append(f"Average components for 99% variance: {avg_components_99:.1f}")
        report.append("")

        report.append("INTERPRETATION")
        report.append("-"*80)
        if avg_consecutive_sim > 0.999:
            report.append("✓ Frames are HIGHLY similar (>99.9% similarity)")
            report.append("  → Frames are likely smoothly interpolated")
        elif avg_consecutive_sim > 0.99:
            report.append("✓ Frames are very similar (>99% similarity)")
            report.append("  → Some interpolation, but some variation")
        else:
            report.append("✓ Frames have moderate variation")
            report.append("  → Frames may be independently learned")

        if avg_components_95 < 10:
            report.append(f"✓ Only ~{avg_components_95:.0f} components explain 95% variance")
            report.append("  → Effective dimensionality is LOW")
            report.append("  → Most frames are redundant")
        else:
            report.append(f"✓ ~{avg_components_95:.0f} components needed for 95% variance")
            report.append("  → Higher effective dimensionality")

        report.append("")
        report.append("RECOMMENDED EXPLORATION STRATEGY")
        report.append("-"*80)
        if avg_consecutive_sim > 0.999 and avg_components_95 < 5:
            report.append("Strategy: SINGLE FRAME (frames are nearly identical)")
            report.append(f"  - Use frame index ~50-100 (typical text length)")
            report.append(f"  - Explore 256 dimensions (or focus on prosody: 128 dims)")
        else:
            report.append("Strategy: MULTI-FRAME with key indices")
            report.append(f"  - Explore frames: [0, 10, 50, 200, 509]")
            report.append(f"  - Each frame: 256 dimensions")

        report.append("")
        report.append("="*80)

        report_text = "\n".join(report)
        with open(self.output_dir / 'frame_variation_report.txt', 'w') as f:
            f.write(report_text)

        print(f"\n{report_text}")
        print(f"\nSaved: {self.output_dir / 'frame_variation_report.txt'}")

    def run_analysis(self):
        """Run complete frame variation analysis."""
        print("="*80)
        print("FRAME VARIATION ANALYSIS")
        print("="*80)

        self.load_voices()

        all_results = self.analyze_all_voices()
        self.analyze_frame_differences()
        self.analyze_decoder_vs_prosody_variation()
        self.analyze_unique_frames()
        self.generate_report(all_results)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)


def main():
    analyzer = FrameVariationAnalyzer()
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
