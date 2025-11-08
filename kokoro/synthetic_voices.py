"""
Synthetic Voice Generator for Kokoro TTS

Generates random synthetic voices using distribution-aware sampling
with flattened distributions for more expressivity.
"""

import torch
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import random
import string


class SyntheticVoiceGenerator:
    """
    Generate random synthetic voices using PCA-based manipulation.

    Features:
    - Distribution-aware random sampling
    - Flattened distributions for increased expressivity
    - Configurable exaggeration (default 2x for random generation)
    - Automatic name generation
    """

    def __init__(
        self,
        pca_system,
        max_exaggeration: float = 2.0,
        flatten_factor: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialize the Synthetic Voice Generator.

        Args:
            pca_system: VoicePCASystem instance (must be fitted)
            max_exaggeration: Maximum exaggeration for random voices (default 2x)
            flatten_factor: How much to flatten the distribution (0=none, 1=fully flat)
                           Higher values increase expressivity by boosting low-variance components
            seed: Random seed for reproducibility
        """
        self.pca_system = pca_system
        self.max_exaggeration = max_exaggeration
        self.flatten_factor = flatten_factor

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Validate PCA system
        if pca_system.pca_model is None or pca_system.centroid is None:
            raise ValueError("PCA system must be fitted before use")

        # Calculate distribution-aware bounds
        self._calculate_sampling_distribution()

    def _calculate_sampling_distribution(self) -> None:
        """
        Calculate flattened distribution for sampling.

        This reduces the influence of high-variance components and boosts
        low-variance components for more diverse voice generation.
        """
        if self.pca_system.pca_bounds is None:
            self.pca_system.calculate_pca_bounds(max_exaggeration=self.max_exaggeration)

        # Get component statistics
        std = self.pca_system.pca_bounds['std']
        data_min = self.pca_system.pca_bounds['data_min']
        data_max = self.pca_system.pca_bounds['data_max']

        # Calculate flattened standard deviations
        # Reduce high-variance, increase low-variance
        std_normalized = std / std.max()
        flatten_weights = 1.0 - self.flatten_factor * (std_normalized - 0.5) * 2

        self.sampling_std = std * flatten_weights

        # Calculate sampling bounds (with exaggeration)
        self.sampling_min = data_min * self.max_exaggeration
        self.sampling_max = data_max * self.max_exaggeration

        # Calculate distribution-aware probability weights
        # Components closer to the center are sampled more conservatively
        self.component_weights = self._calculate_component_weights()

        print(f"Sampling distribution calculated:")
        print(f"  Flattening factor: {self.flatten_factor}")
        print(f"  Max exaggeration: {self.max_exaggeration}")
        print(f"  Original std range: [{std.min():.4f}, {std.max():.4f}]")
        print(f"  Flattened std range: [{self.sampling_std.min():.4f}, {self.sampling_std.max():.4f}]")

    def _calculate_component_weights(self) -> np.ndarray:
        """
        Calculate sampling weights for each component.

        Components with higher variance get slightly reduced sampling range
        to create more balanced voices.
        """
        explained_var = self.pca_system.explained_variance_ratio

        # Inverse variance weighting (slight)
        weights = 1.0 / (1.0 + explained_var * 5)
        weights = weights / weights.sum() * len(weights)  # Normalize to mean=1

        return weights

    def generate_random_pca_coords(
        self,
        distribution: str = 'normal',
        tail_reduction: float = 0.7
    ) -> np.ndarray:
        """
        Generate random PCA coordinates using distribution-aware sampling.

        Args:
            distribution: Sampling distribution ('normal', 'uniform', 'truncated_normal')
            tail_reduction: Factor to reduce probability of extreme values (0-1)
                           Higher values = more conservative sampling

        Returns:
            Random PCA coordinates array
        """
        n_components = self.pca_system.n_components
        pca_coords = np.zeros(n_components)

        for i in range(n_components):
            if distribution == 'normal':
                # Sample from normal distribution with flattened std
                sample = np.random.normal(0, self.sampling_std[i])

                # Apply tail reduction (push extreme values back toward bounds)
                max_val = max(abs(self.sampling_min[i]), abs(self.sampling_max[i]))
                if abs(sample) > max_val:
                    # Exponentially reduce probability of going beyond max
                    excess = abs(sample) - max_val
                    reduction = np.exp(-excess * tail_reduction * 2)
                    sample = np.sign(sample) * (max_val + excess * reduction)

            elif distribution == 'uniform':
                # Uniform sampling within bounds
                sample = np.random.uniform(
                    self.sampling_min[i],
                    self.sampling_max[i]
                )

            elif distribution == 'truncated_normal':
                # Truncated normal within bounds
                sample = np.random.normal(0, self.sampling_std[i])
                sample = np.clip(sample, self.sampling_min[i], self.sampling_max[i])

            else:
                raise ValueError(f"Unknown distribution: {distribution}")

            # Apply component weight
            sample *= self.component_weights[i]

            # Final clipping to bounds
            pca_coords[i] = np.clip(sample, self.sampling_min[i], self.sampling_max[i])

        return pca_coords

    def generate_random_voice(
        self,
        name: Optional[str] = None,
        distribution: str = 'normal',
        tail_reduction: float = 0.7,
        seq_len: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Generate a random synthetic voice.

        Args:
            name: Voice name (auto-generated if None)
            distribution: Sampling distribution type
            tail_reduction: Tail reduction factor
            seq_len: Target sequence length

        Returns:
            Dictionary with 'name', 'voice_tensor', 'pca_coords'
        """
        # Generate PCA coordinates
        pca_coords = self.generate_random_pca_coords(
            distribution=distribution,
            tail_reduction=tail_reduction
        )

        # Transform to voice space
        voice_tensor = self.pca_system.inverse_pca_transform(pca_coords, seq_len=seq_len)

        # Generate name if not provided
        if name is None:
            name = self.generate_voice_name()

        return {
            'name': name,
            'voice_tensor': voice_tensor,
            'pca_coords': pca_coords,
            'distribution': distribution,
            'tail_reduction': tail_reduction
        }

    def generate_voice_name(self, prefix: str = 'syn') -> str:
        """
        Generate a random voice name.

        Args:
            prefix: Prefix for the name (default 'syn' for synthetic)

        Returns:
            Random voice name like 'syn_aurora', 'syn_zephyr', etc.
        """
        # Curated list of pleasant-sounding names
        names = [
            'aurora', 'blaze', 'cascade', 'delta', 'echo', 'flux', 'galaxy',
            'harmony', 'iris', 'jade', 'kilo', 'luna', 'matrix', 'nebula',
            'onyx', 'phoenix', 'quartz', 'raven', 'stellar', 'titan',
            'umbra', 'vertex', 'whisper', 'xenon', 'yarn', 'zephyr',
            'axel', 'beacon', 'cipher', 'drift', 'ember', 'frost',
            'glimmer', 'halo', 'indigo', 'jazz', 'karma', 'lyric',
            'mystique', 'nova', 'oracle', 'pulse', 'quest', 'rhythm',
            'sage', 'tempo', 'unity', 'vibe', 'wave', 'zeal'
        ]

        # Add random suffix for uniqueness
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        name = f"{prefix}_{random.choice(names)}_{suffix}"

        return name

    def generate_batch(
        self,
        n_voices: int = 10,
        distribution: str = 'normal',
        tail_reduction: float = 0.7,
        seq_len: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate a batch of random synthetic voices.

        Args:
            n_voices: Number of voices to generate
            distribution: Sampling distribution type
            tail_reduction: Tail reduction factor
            seq_len: Target sequence length
            save_dir: Optional directory to save voices

        Returns:
            List of voice dictionaries
        """
        voices = []

        print(f"Generating {n_voices} synthetic voices...")

        for i in range(n_voices):
            voice = self.generate_random_voice(
                distribution=distribution,
                tail_reduction=tail_reduction,
                seq_len=seq_len
            )
            voices.append(voice)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_voices} voices")

        print(f"Generated {len(voices)} synthetic voices")

        # Save if requested
        if save_dir:
            self.save_voices(voices, save_dir)

        return voices

    def save_voices(self, voices: List[Dict], save_dir: str) -> None:
        """
        Save synthetic voices to disk.

        Args:
            voices: List of voice dictionaries
            save_dir: Directory to save voices
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for voice in voices:
            name = voice['name']
            voice_tensor = voice['voice_tensor']

            # Save voice tensor
            torch.save(voice_tensor, save_path / f"{name}.pt")

            # Save metadata
            metadata = {
                'name': name,
                'pca_coords': voice['pca_coords'].tolist(),
                'distribution': voice['distribution'],
                'tail_reduction': voice['tail_reduction'],
                'shape': list(voice_tensor.shape)
            }

            import json
            with open(save_path / f"{name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"Saved {len(voices)} voices to {save_path}")

    def get_sampling_statistics(self) -> Dict:
        """Get statistics about the sampling distribution."""
        return {
            'n_components': self.pca_system.n_components,
            'max_exaggeration': self.max_exaggeration,
            'flatten_factor': self.flatten_factor,
            'sampling_std_range': [
                float(self.sampling_std.min()),
                float(self.sampling_std.max())
            ],
            'sampling_bounds': {
                'min': self.sampling_min.tolist(),
                'max': self.sampling_max.tolist()
            },
            'component_weights_range': [
                float(self.component_weights.min()),
                float(self.component_weights.max())
            ]
        }


class VoiceNameGenerator:
    """
    Advanced voice name generator with different styles.
    """

    FANTASY_NAMES = [
        'aeliana', 'brynn', 'celestia', 'darian', 'elara', 'finnian',
        'galadriel', 'haven', 'isolde', 'jareth', 'kaelum', 'liora',
        'meridian', 'nyx', 'oberon', 'petra', 'quillon', 'rielle',
        'soren', 'thalia', 'ulric', 'vesper', 'wynter', 'xara', 'ysolde', 'zara'
    ]

    TECH_NAMES = [
        'alpha', 'beta', 'cipher', 'delta', 'epsilon', 'flux',
        'gamma', 'helix', 'ion', 'junction', 'krypton', 'lambda',
        'matrix', 'neuron', 'omega', 'pixel', 'quantum', 'radix',
        'sigma', 'tesla', 'ultra', 'vector', 'watt', 'xenon', 'yield', 'zenith'
    ]

    NATURE_NAMES = [
        'aspen', 'birch', 'cedar', 'dawn', 'ember', 'fern',
        'grove', 'hazel', 'ivy', 'jade', 'kelp', 'leaf',
        'moss', 'nectar', 'oak', 'petal', 'quill', 'river',
        'sage', 'thorn', 'umber', 'valley', 'willow', 'xylem', 'yarrow', 'zinnia'
    ]

    @classmethod
    def generate(cls, style: str = 'random', prefix: str = 'syn') -> str:
        """
        Generate a voice name with specified style.

        Args:
            style: Name style ('fantasy', 'tech', 'nature', 'random')
            prefix: Prefix for the name

        Returns:
            Generated name
        """
        if style == 'fantasy':
            name_list = cls.FANTASY_NAMES
        elif style == 'tech':
            name_list = cls.TECH_NAMES
        elif style == 'nature':
            name_list = cls.NATURE_NAMES
        else:
            # Random mix
            name_list = cls.FANTASY_NAMES + cls.TECH_NAMES + cls.NATURE_NAMES

        base_name = random.choice(name_list)
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))

        return f"{prefix}_{base_name}_{suffix}"
