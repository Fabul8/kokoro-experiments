#!/usr/bin/env python3
"""
Comprehensive voice embedding analysis for Kokoro TTS.

This script performs:
1. PCA analysis on voice embeddings
2. Centroid calculation for all voices and voice groups
3. Inter-layer correlation analysis
4. Visualization generation

Voice embeddings in Kokoro have shape [seq_len, 256] where:
- [:, :128]: Style embedding for decoder
- [:, 128:]: Style embedding for prosody predictor
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from typing import Dict, List, Tuple
import json

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)


class VoiceAnalyzer:
    """Analyzer for Kokoro voice embeddings."""

    def __init__(self, voices_dir: str = 'voices'):
        self.voices_dir = Path(voices_dir)
        self.voices: Dict[str, torch.Tensor] = {}
        self.analysis_dir = self.voices_dir / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def load_voices(self, pattern: str = '*.pt') -> int:
        """Load all voice files matching the pattern."""
        voice_files = list(self.voices_dir.glob(pattern))

        print(f"Loading {len(voice_files)} voice files from {self.voices_dir}")

        for voice_file in sorted(voice_files):
            voice_name = voice_file.stem
            try:
                voice_tensor = torch.load(voice_file, map_location='cpu', weights_only=True)
                self.voices[voice_name] = voice_tensor
                print(f"  ✓ {voice_name}: shape {voice_tensor.shape}")
            except Exception as e:
                print(f"  ✗ {voice_name}: {e}")

        print(f"\nLoaded {len(self.voices)} voices")
        return len(self.voices)

    def get_voice_groups(self) -> Dict[str, List[str]]:
        """Group voices by language/accent prefix."""
        groups = {}
        for voice_name in self.voices.keys():
            prefix = voice_name[:2]  # e.g., 'af', 'am', 'bf', etc.
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(voice_name)
        return groups

    def extract_embeddings(self, layer: str = 'both') -> Tuple[np.ndarray, List[str]]:
        """
        Extract embeddings from all voices.

        Args:
            layer: Which layer to extract ('decoder', 'prosody', or 'both')

        Returns:
            embeddings: Array of shape (n_voices, embedding_dim)
            voice_names: List of voice names corresponding to embeddings
        """
        embeddings_list = []
        voice_names = []

        for voice_name, voice_tensor in sorted(self.voices.items()):
            # Voice tensor shape: [seq_len, 1, 256] or [seq_len, 256]
            # Remove extra dimensions and average over sequence length
            voice_tensor = voice_tensor.squeeze()  # Remove singleton dimensions
            avg_embedding = voice_tensor.mean(dim=0).numpy()

            if layer == 'decoder':
                embedding = avg_embedding[:128]
            elif layer == 'prosody':
                embedding = avg_embedding[128:]
            else:  # both
                embedding = avg_embedding

            embeddings_list.append(embedding)
            voice_names.append(voice_name)

        embeddings = np.array(embeddings_list)
        print(f"\nExtracted embeddings: {embeddings.shape}")
        return embeddings, voice_names

    def compute_centroids(self) -> Dict[str, np.ndarray]:
        """Compute centroids for each voice group and overall."""
        embeddings, voice_names = self.extract_embeddings('both')
        groups = self.get_voice_groups()

        centroids = {}

        # Overall centroid
        centroids['overall'] = embeddings.mean(axis=0)

        # Group centroids
        for group_name, group_voices in groups.items():
            group_indices = [i for i, v in enumerate(voice_names) if v in group_voices]
            if group_indices:
                centroids[group_name] = embeddings[group_indices].mean(axis=0)

        # Save centroids
        centroid_file = self.analysis_dir / 'centroids.npz'
        np.savez(centroid_file, **centroids)
        print(f"\nSaved centroids to {centroid_file}")

        # Save centroid statistics
        stats = {
            'overall': {
                'mean': float(centroids['overall'].mean()),
                'std': float(centroids['overall'].std()),
                'min': float(centroids['overall'].min()),
                'max': float(centroids['overall'].max()),
            }
        }

        for group_name, centroid in centroids.items():
            if group_name != 'overall':
                stats[group_name] = {
                    'mean': float(centroid.mean()),
                    'std': float(centroid.std()),
                    'distance_from_overall': float(np.linalg.norm(centroid - centroids['overall'])),
                }

        with open(self.analysis_dir / 'centroid_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return centroids

    def pca_analysis(self, n_components: int = 50):
        """Perform PCA analysis on voice embeddings."""
        print("\n" + "="*60)
        print("PCA Analysis")
        print("="*60)

        results = {}

        for layer in ['decoder', 'prosody', 'both']:
            print(f"\nAnalyzing {layer} layer...")
            embeddings, voice_names = self.extract_embeddings(layer)

            # Perform PCA
            pca = PCA(n_components=min(n_components, embeddings.shape[1]))
            transformed = pca.fit_transform(embeddings)

            results[layer] = {
                'pca': pca,
                'transformed': transformed,
                'voice_names': voice_names,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            }

            print(f"  PC1 explains: {pca.explained_variance_ratio_[0]:.2%}")
            print(f"  PC1-5 explain: {pca.explained_variance_ratio_[:5].sum():.2%}")
            print(f"  PC1-10 explain: {pca.explained_variance_ratio_[:10].sum():.2%}")

        return results

    def inter_layer_correlation(self):
        """Analyze correlations between decoder and prosody layers."""
        print("\n" + "="*60)
        print("Inter-Layer Correlation Analysis")
        print("="*60)

        decoder_embs, voice_names = self.extract_embeddings('decoder')
        prosody_embs, _ = self.extract_embeddings('prosody')

        # Compute correlation matrix between decoder and prosody dimensions
        correlation_matrix = np.corrcoef(decoder_embs.T, prosody_embs.T)

        # Extract the cross-correlation block
        cross_corr = correlation_matrix[:128, 128:]

        print(f"\nCross-correlation matrix shape: {cross_corr.shape}")
        print(f"Mean absolute correlation: {np.abs(cross_corr).mean():.4f}")
        print(f"Max correlation: {np.abs(cross_corr).max():.4f}")

        # Find most correlated dimension pairs
        flat_indices = np.argsort(np.abs(cross_corr).flatten())[::-1]
        top_k = 10
        print(f"\nTop {top_k} correlated dimension pairs:")
        for i in range(top_k):
            idx = flat_indices[i]
            dec_dim = idx // 128
            pros_dim = idx % 128
            corr_val = cross_corr[dec_dim, pros_dim]
            print(f"  Decoder[{dec_dim}] ↔ Prosody[{pros_dim}]: {corr_val:+.4f}")

        return {
            'cross_correlation': cross_corr,
            'decoder_embeddings': decoder_embs,
            'prosody_embeddings': prosody_embs,
            'voice_names': voice_names,
        }

    def plot_pca_variance(self, pca_results: Dict):
        """Plot explained variance for each PCA analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, layer in enumerate(['decoder', 'prosody', 'both']):
            ax = axes[idx]
            result = pca_results[layer]

            # Plot explained variance
            ax.bar(range(1, len(result['explained_variance_ratio']) + 1),
                   result['explained_variance_ratio'],
                   alpha=0.6, label='Individual')

            ax.plot(range(1, len(result['cumulative_variance']) + 1),
                    result['cumulative_variance'],
                    'r-', linewidth=2, label='Cumulative')

            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title(f'PCA Explained Variance - {layer.capitalize()} Layer')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'pca_variance.png', bbox_inches='tight')
        print(f"\nSaved PCA variance plot to {self.analysis_dir / 'pca_variance.png'}")
        plt.close()

    def plot_pca_2d(self, pca_results: Dict):
        """Plot 2D PCA projections colored by voice groups."""
        groups = self.get_voice_groups()

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, layer in enumerate(['decoder', 'prosody', 'both']):
            ax = axes[idx]
            result = pca_results[layer]
            transformed = result['transformed']
            voice_names = result['voice_names']

            # Color by group
            colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
            group_to_color = {g: colors[i] for i, g in enumerate(sorted(groups.keys()))}

            for voice_name, (pc1, pc2) in zip(voice_names, transformed[:, :2]):
                group = voice_name[:2]
                ax.scatter(pc1, pc2, c=[group_to_color[group]], alpha=0.7, s=100)

            ax.set_xlabel(f'PC1 ({result["explained_variance_ratio"][0]:.1%})')
            ax.set_ylabel(f'PC2 ({result["explained_variance_ratio"][1]:.1%})')
            ax.set_title(f'{layer.capitalize()} Layer PCA')
            ax.grid(True, alpha=0.3)

        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=group_to_color[g], markersize=8, label=g)
                   for g in sorted(groups.keys())]
        fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.05, 0.5))

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'pca_2d.png', bbox_inches='tight')
        print(f"Saved 2D PCA plot to {self.analysis_dir / 'pca_2d.png'}")
        plt.close()

    def plot_tsne(self, layer: str = 'both', perplexity: int = 30):
        """Plot t-SNE visualization."""
        print(f"\nGenerating t-SNE visualization for {layer} layer...")

        embeddings, voice_names = self.extract_embeddings(layer)
        groups = self.get_voice_groups()

        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        transformed = tsne.fit_transform(embeddings)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))

        colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
        group_to_color = {g: colors[i] for i, g in enumerate(sorted(groups.keys()))}

        for voice_name, (x, y) in zip(voice_names, transformed):
            group = voice_name[:2]
            ax.scatter(x, y, c=[group_to_color[group]], alpha=0.7, s=100)
            ax.annotate(voice_name, (x, y), fontsize=6, alpha=0.7)

        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE Visualization - {layer.capitalize()} Layer')
        ax.grid(True, alpha=0.3)

        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=group_to_color[g], markersize=8, label=g)
                   for g in sorted(groups.keys())]
        ax.legend(handles=handles, loc='best')

        plt.tight_layout()
        plt.savefig(self.analysis_dir / f'tsne_{layer}.png', bbox_inches='tight')
        print(f"Saved t-SNE plot to {self.analysis_dir / f'tsne_{layer}.png'}")
        plt.close()

    def plot_correlation_heatmap(self, corr_results: Dict):
        """Plot inter-layer correlation heatmap."""
        cross_corr = corr_results['cross_correlation']

        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(cross_corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    xticklabels=20, yticklabels=20, cbar_kws={'label': 'Correlation'})

        ax.set_xlabel('Prosody Layer Dimensions')
        ax.set_ylabel('Decoder Layer Dimensions')
        ax.set_title('Inter-Layer Correlation: Decoder vs Prosody')

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'interlayer_correlation.png', bbox_inches='tight')
        print(f"Saved correlation heatmap to {self.analysis_dir / 'interlayer_correlation.png'}")
        plt.close()

    def plot_embedding_distributions(self):
        """Plot distribution of embedding values for each layer."""
        decoder_embs, _ = self.extract_embeddings('decoder')
        prosody_embs, _ = self.extract_embeddings('prosody')

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Decoder layer
        axes[0].hist(decoder_embs.flatten(), bins=100, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Embedding Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Decoder Layer Embedding Distribution')
        axes[0].axvline(decoder_embs.mean(), color='r', linestyle='--',
                        label=f'Mean: {decoder_embs.mean():.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Prosody layer
        axes[1].hist(prosody_embs.flatten(), bins=100, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_xlabel('Embedding Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Prosody Layer Embedding Distribution')
        axes[1].axvline(prosody_embs.mean(), color='r', linestyle='--',
                        label=f'Mean: {prosody_embs.mean():.3f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'embedding_distributions.png', bbox_inches='tight')
        print(f"Saved embedding distributions to {self.analysis_dir / 'embedding_distributions.png'}")
        plt.close()

    def plot_dimension_variance(self):
        """Plot variance across voices for each dimension."""
        decoder_embs, _ = self.extract_embeddings('decoder')
        prosody_embs, _ = self.extract_embeddings('prosody')

        decoder_var = decoder_embs.var(axis=0)
        prosody_var = prosody_embs.var(axis=0)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Decoder variance
        axes[0].bar(range(len(decoder_var)), decoder_var, alpha=0.7)
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('Variance')
        axes[0].set_title('Decoder Layer: Variance Across Voices per Dimension')
        axes[0].grid(True, alpha=0.3)

        # Mark top 10 dimensions
        top_10_dec = np.argsort(decoder_var)[-10:]
        axes[0].bar(top_10_dec, decoder_var[top_10_dec], alpha=0.9, color='red',
                    label='Top 10 Most Variable')
        axes[0].legend()

        # Prosody variance
        axes[1].bar(range(len(prosody_var)), prosody_var, alpha=0.7, color='orange')
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('Variance')
        axes[1].set_title('Prosody Layer: Variance Across Voices per Dimension')
        axes[1].grid(True, alpha=0.3)

        # Mark top 10 dimensions
        top_10_pros = np.argsort(prosody_var)[-10:]
        axes[1].bar(top_10_pros, prosody_var[top_10_pros], alpha=0.9, color='darkred',
                    label='Top 10 Most Variable')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'dimension_variance.png', bbox_inches='tight')
        print(f"Saved dimension variance plot to {self.analysis_dir / 'dimension_variance.png'}")
        plt.close()

        # Save top dimensions
        top_dims = {
            'decoder_top_10': top_10_dec.tolist(),
            'decoder_variances': decoder_var[top_10_dec].tolist(),
            'prosody_top_10': top_10_pros.tolist(),
            'prosody_variances': prosody_var[top_10_pros].tolist(),
        }
        with open(self.analysis_dir / 'top_variable_dimensions.json', 'w') as f:
            json.dump(top_dims, f, indent=2)

    def generate_summary_report(self, pca_results: Dict, corr_results: Dict, centroids: Dict):
        """Generate a text summary report of the analysis."""
        report = []
        report.append("="*80)
        report.append("KOKORO VOICE ANALYSIS REPORT")
        report.append("="*80)
        report.append("")

        # Voice statistics
        report.append("VOICE STATISTICS")
        report.append("-"*80)
        report.append(f"Total voices analyzed: {len(self.voices)}")
        groups = self.get_voice_groups()
        report.append(f"Voice groups: {len(groups)}")
        for group, voices in sorted(groups.items()):
            report.append(f"  {group}: {len(voices)} voices")
        report.append("")

        # PCA results
        report.append("PCA ANALYSIS")
        report.append("-"*80)
        for layer in ['decoder', 'prosody', 'both']:
            result = pca_results[layer]
            report.append(f"\n{layer.upper()} Layer:")
            report.append(f"  PC1 variance explained: {result['explained_variance_ratio'][0]:.2%}")
            report.append(f"  PC1-5 cumulative: {result['cumulative_variance'][4]:.2%}")
            report.append(f"  PC1-10 cumulative: {result['cumulative_variance'][9]:.2%}")
        report.append("")

        # Inter-layer correlation
        report.append("INTER-LAYER CORRELATION")
        report.append("-"*80)
        cross_corr = corr_results['cross_correlation']
        report.append(f"Mean absolute correlation: {np.abs(cross_corr).mean():.4f}")
        report.append(f"Max absolute correlation: {np.abs(cross_corr).max():.4f}")
        report.append(f"Correlation > 0.5: {(np.abs(cross_corr) > 0.5).sum()} pairs")
        report.append(f"Correlation > 0.7: {(np.abs(cross_corr) > 0.7).sum()} pairs")
        report.append("")

        # Centroid analysis
        report.append("CENTROID ANALYSIS")
        report.append("-"*80)
        report.append(f"Overall centroid norm: {np.linalg.norm(centroids['overall']):.4f}")
        report.append("\nGroup distances from overall centroid:")
        for group in sorted(groups.keys()):
            if group in centroids:
                dist = np.linalg.norm(centroids[group] - centroids['overall'])
                report.append(f"  {group}: {dist:.4f}")
        report.append("")

        # Most variable dimensions
        decoder_embs, _ = self.extract_embeddings('decoder')
        prosody_embs, _ = self.extract_embeddings('prosody')
        decoder_var = decoder_embs.var(axis=0)
        prosody_var = prosody_embs.var(axis=0)

        report.append("MOST VARIABLE DIMENSIONS")
        report.append("-"*80)
        report.append("Decoder layer (top 5):")
        top_5_dec = np.argsort(decoder_var)[-5:][::-1]
        for i, dim in enumerate(top_5_dec):
            report.append(f"  {i+1}. Dimension {dim}: variance = {decoder_var[dim]:.6f}")

        report.append("\nProsody layer (top 5):")
        top_5_pros = np.argsort(prosody_var)[-5:][::-1]
        for i, dim in enumerate(top_5_pros):
            report.append(f"  {i+1}. Dimension {dim}: variance = {prosody_var[dim]:.6f}")
        report.append("")

        report.append("="*80)

        # Save report
        report_text = "\n".join(report)
        report_file = self.analysis_dir / 'analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)

        print(f"\n{report_text}")
        print(f"\nSaved report to {report_file}")

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("="*80)
        print("KOKORO VOICE EMBEDDING ANALYSIS")
        print("="*80)

        # Load voices
        if not self.load_voices():
            print("No voices found! Please download voices first.")
            return

        # Compute centroids
        centroids = self.compute_centroids()

        # PCA analysis
        pca_results = self.pca_analysis(n_components=50)

        # Inter-layer correlation
        corr_results = self.inter_layer_correlation()

        # Generate visualizations
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)

        self.plot_pca_variance(pca_results)
        self.plot_pca_2d(pca_results)
        self.plot_tsne('both')
        self.plot_correlation_heatmap(corr_results)
        self.plot_embedding_distributions()
        self.plot_dimension_variance()

        # Generate summary report
        self.generate_summary_report(pca_results, corr_results, centroids)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"All results saved to: {self.analysis_dir.absolute()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Kokoro voice embeddings')
    parser.add_argument('--voices-dir', default='voices',
                        help='Directory containing voice .pt files')
    parser.add_argument('--pattern', default='*.pt',
                        help='Pattern to match voice files')

    args = parser.parse_args()

    analyzer = VoiceAnalyzer(args.voices_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
