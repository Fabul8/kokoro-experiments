#!/usr/bin/env python3
"""
Incremental PCA Analysis for Kokoro Voices.

This script performs incremental PCA by adding voices one at a time and tracking:
1. How many dimensions are needed to reach 99.99% explained variance
2. How the intrinsic dimensionality changes as more voices are added
3. Whether diversity increases with more voices from different languages

The voices are shuffled before analysis to ensure balanced sampling across languages.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, List, Tuple
import json
import random

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class IncrementalPCAAnalyzer:
    """Incremental PCA analyzer for voice embeddings."""

    def __init__(self, voices_dir: str = 'voices', output_dir: str = 'voices/analysis'):
        self.voices_dir = Path(voices_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voices: Dict[str, torch.Tensor] = {}

    def load_voices(self) -> int:
        """Load all voice files."""
        voice_files = list(self.voices_dir.glob('*.pt'))

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
            prefix = voice_name[:2]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(voice_name)
        return groups

    def shuffle_voices_balanced(self) -> List[str]:
        """
        Shuffle voices ensuring balanced sampling across language groups.

        Strategy: Round-robin through language groups to ensure diversity
        throughout the incremental analysis.
        """
        groups = self.get_voice_groups()

        # Create list of (group, voice) pairs
        group_voices = []
        for group, voices in groups.items():
            # Shuffle within each group
            shuffled = list(voices)
            random.shuffle(shuffled)
            for voice in shuffled:
                group_voices.append((group, voice))

        # Sort by group to enable round-robin
        group_voices.sort(key=lambda x: x[0])

        # Extract voices in round-robin fashion across groups
        voices_by_group = {}
        for group, voice in group_voices:
            if group not in voices_by_group:
                voices_by_group[group] = []
            voices_by_group[group].append(voice)

        # Round-robin selection
        balanced_order = []
        max_group_size = max(len(v) for v in voices_by_group.values())

        for i in range(max_group_size):
            for group in sorted(voices_by_group.keys()):
                if i < len(voices_by_group[group]):
                    balanced_order.append(voices_by_group[group][i])

        print(f"\nBalanced shuffle order (first 10): {balanced_order[:10]}")
        print(f"Language distribution in first 20:")
        first_20_groups = [v[:2] for v in balanced_order[:20]]
        for group in sorted(set(first_20_groups)):
            count = first_20_groups.count(group)
            print(f"  {group}: {count}")

        return balanced_order

    def extract_embedding(self, voice_name: str, layer: str = 'both') -> np.ndarray:
        """
        Extract embedding from a single voice.

        Args:
            voice_name: Name of the voice
            layer: Which layer to extract ('decoder', 'prosody', or 'both')

        Returns:
            embedding: 1D array of embedding values
        """
        voice_tensor = self.voices[voice_name]
        # Remove extra dimensions and average over sequence length
        voice_tensor = voice_tensor.squeeze()  # Remove singleton dimensions
        avg_embedding = voice_tensor.mean(dim=0).numpy()

        if layer == 'decoder':
            return avg_embedding[:128]
        elif layer == 'prosody':
            return avg_embedding[128:]
        else:  # both
            return avg_embedding

    def incremental_pca_analysis(
        self,
        voice_order: List[str],
        layer: str = 'both',
        target_variance: float = 0.9999
    ) -> Dict:
        """
        Perform incremental PCA analysis.

        Args:
            voice_order: Order in which to add voices
            layer: Which layer to analyze
            target_variance: Target explained variance (default: 99.99%)

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*80}")
        print(f"Incremental PCA Analysis - {layer.upper()} layer")
        print(f"Target variance: {target_variance:.2%}")
        print(f"{'='*80}\n")

        results = {
            'layer': layer,
            'target_variance': target_variance,
            'voice_order': voice_order,
            'incremental_results': [],
        }

        embeddings_accumulated = []

        for i, voice_name in enumerate(voice_order, 1):
            # Add new voice embedding
            embedding = self.extract_embedding(voice_name, layer)
            embeddings_accumulated.append(embedding)

            # Stack all accumulated embeddings
            X = np.array(embeddings_accumulated)

            # Perform PCA
            n_samples, n_features = X.shape
            max_components = min(n_samples, n_features)

            pca = PCA(n_components=max_components)
            pca.fit(X)

            # Find number of components for target variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components_needed = np.searchsorted(cumulative_variance, target_variance) + 1

            # Clamp to max possible
            n_components_needed = min(n_components_needed, max_components)

            # Actual variance explained with these components
            actual_variance = cumulative_variance[n_components_needed - 1] if n_components_needed > 0 else 0.0

            result = {
                'num_voices': i,
                'voice_name': voice_name,
                'voice_group': voice_name[:2],
                'n_components_needed': int(n_components_needed),
                'actual_variance_explained': float(actual_variance),
                'total_variance': float(pca.explained_variance_.sum()),
                'first_pc_variance': float(pca.explained_variance_ratio_[0]),
                'max_possible_components': int(max_components),
            }

            results['incremental_results'].append(result)

            if i % 10 == 0 or i == len(voice_order):
                print(f"Voice {i:3d} ({voice_name:20s}): "
                      f"{n_components_needed:3d} dims for {actual_variance:.4%} variance "
                      f"(max possible: {max_components})")

        return results

    def plot_incremental_pca(self, results: Dict):
        """Plot incremental PCA results."""
        layer = results['layer']
        incremental = results['incremental_results']

        num_voices = [r['num_voices'] for r in incremental]
        n_components = [r['n_components_needed'] for r in incremental]
        variance_explained = [r['actual_variance_explained'] for r in incremental]
        first_pc_variance = [r['first_pc_variance'] for r in incremental]
        max_possible = [r['max_possible_components'] for r in incremental]

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Plot 1: Components needed vs number of voices
        ax = axes[0, 0]
        ax.plot(num_voices, n_components, 'b-', linewidth=2, label='Components needed')
        ax.plot(num_voices, max_possible, 'r--', linewidth=1, alpha=0.5, label='Max possible')
        ax.fill_between(num_voices, n_components, alpha=0.3)
        ax.set_xlabel('Number of Voices')
        ax.set_ylabel('Components Needed')
        ax.set_title(f'Incremental PCA: Components for 99.99% Variance\n{layer.capitalize()} Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Variance explained
        ax = axes[0, 1]
        ax.plot(num_voices, variance_explained, 'g-', linewidth=2)
        ax.axhline(y=results['target_variance'], color='r', linestyle='--',
                   label=f'Target: {results["target_variance"]:.2%}')
        ax.set_xlabel('Number of Voices')
        ax.set_ylabel('Variance Explained')
        ax.set_title('Actual Variance Explained')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.99, 1.0])

        # Plot 3: First PC variance
        ax = axes[1, 0]
        ax.plot(num_voices, first_pc_variance, 'purple', linewidth=2)
        ax.set_xlabel('Number of Voices')
        ax.set_ylabel('First PC Variance Ratio')
        ax.set_title('First Principal Component Variance\n(Decreases as diversity increases)')
        ax.grid(True, alpha=0.3)

        # Plot 4: Components needed (rate of change)
        ax = axes[1, 1]
        # Calculate rate of change
        components_diff = np.diff([0] + n_components)
        ax.bar(num_voices, components_diff, alpha=0.7, color='orange')
        ax.set_xlabel('Number of Voices')
        ax.set_ylabel('New Components Added')
        ax.set_title('Incremental Dimensionality Growth')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / f'incremental_pca_{layer}.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"\nSaved plot to {output_file}")
        plt.close()

    def plot_group_contribution(self, results: Dict):
        """Plot how different language groups contribute to dimensionality."""
        incremental = results['incremental_results']
        layer = results['layer']

        # Track when each new component was added and which group caused it
        group_components = {}
        prev_components = 0

        for result in incremental:
            group = result['voice_group']
            n_comp = result['n_components_needed']
            new_components = n_comp - prev_components

            if group not in group_components:
                group_components[group] = 0
            group_components[group] += new_components
            prev_components = n_comp

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        groups = sorted(group_components.keys())
        contributions = [group_components[g] for g in groups]

        bars = ax.bar(groups, contributions, alpha=0.7)

        # Color bars by language family (rough grouping)
        colors = {
            'a': 'blue',    # American English
            'b': 'cyan',    # British English
            'e': 'orange',  # Spanish
            'f': 'red',     # French
            'h': 'green',   # Hindi
            'i': 'yellow',  # Italian
            'j': 'purple',  # Japanese
            'p': 'brown',   # Portuguese
            'z': 'pink',    # Chinese
        }

        for bar, group in zip(bars, groups):
            bar.set_color(colors.get(group[0], 'gray'))

        ax.set_xlabel('Language Group')
        ax.set_ylabel('Total Components Contributed')
        ax.set_title(f'Dimensionality Contribution by Language Group\n{layer.capitalize()} Layer')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.output_dir / f'group_contribution_{layer}.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close()

    def generate_report(self, results_all_layers: Dict):
        """Generate comprehensive report."""
        report = []
        report.append("="*80)
        report.append("INCREMENTAL PCA ANALYSIS REPORT")
        report.append("="*80)
        report.append("")

        for layer, results in results_all_layers.items():
            incremental = results['incremental_results']
            final = incremental[-1]

            report.append(f"\n{layer.upper()} LAYER")
            report.append("-"*80)
            report.append(f"Target variance: {results['target_variance']:.2%}")
            report.append(f"Total voices analyzed: {len(incremental)}")
            report.append(f"\nFinal results (all {final['num_voices']} voices):")
            report.append(f"  Components needed: {final['n_components_needed']}")
            report.append(f"  Variance explained: {final['actual_variance_explained']:.4%}")
            report.append(f"  Max possible components: {final['max_possible_components']}")
            report.append(f"  Utilization: {final['n_components_needed'] / final['max_possible_components']:.2%}")

            # Find plateau point (where components stabilize)
            components_series = [r['n_components_needed'] for r in incremental]

            # Find when we reach 90% of final components
            threshold = 0.9 * final['n_components_needed']
            plateau_idx = next((i for i, c in enumerate(components_series) if c >= threshold), len(components_series) - 1)

            report.append(f"\nPlateau analysis:")
            report.append(f"  90% of final components reached at voice {plateau_idx + 1}")
            report.append(f"  Voice: {incremental[plateau_idx]['voice_name']}")

            # Breakdown by adding groups
            report.append(f"\nFirst 10 voices:")
            for i in range(min(10, len(incremental))):
                r = incremental[i]
                report.append(f"  {i+1:2d}. {r['voice_name']:20s} ({r['voice_group']}): "
                            f"{r['n_components_needed']:3d} components")

            report.append("")

        report.append("="*80)

        # Save report
        report_text = "\n".join(report)
        report_file = self.output_dir / 'incremental_pca_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)

        print(f"\n{report_text}")
        print(f"\nSaved report to {report_file}")

    def run_analysis(self, seed: int = 42):
        """Run complete incremental PCA analysis."""
        random.seed(seed)
        np.random.seed(seed)

        print("="*80)
        print("INCREMENTAL PCA ANALYSIS")
        print("="*80)

        # Load voices
        if not self.load_voices():
            print("No voices found!")
            return

        # Get balanced shuffle order
        voice_order = self.shuffle_voices_balanced()

        # Run analysis for each layer
        results_all = {}

        for layer in ['decoder', 'prosody', 'both']:
            results = self.incremental_pca_analysis(voice_order, layer, target_variance=0.9999)
            results_all[layer] = results

            # Save results
            output_file = self.output_dir / f'incremental_pca_{layer}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {output_file}")

            # Generate plots
            self.plot_incremental_pca(results)
            self.plot_group_contribution(results)

        # Generate comprehensive report
        self.generate_report(results_all)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Incremental PCA analysis for Kokoro voices')
    parser.add_argument('--voices-dir', default='voices', help='Directory containing voice files')
    parser.add_argument('--output-dir', default='voices/analysis', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    analyzer = IncrementalPCAAnalyzer(args.voices_dir, args.output_dir)
    analyzer.run_analysis(seed=args.seed)


if __name__ == '__main__':
    main()
