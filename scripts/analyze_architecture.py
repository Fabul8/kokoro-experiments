#!/usr/bin/env python3
"""
Analyze the Kokoro TTS model architecture.

This script:
1. Loads the model and extracts layer information
2. Computes parameter counts for each module
3. Analyzes layer dimensions and connectivity
4. Generates architecture diagrams and reports
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Dict, List, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro import KModel


class ArchitectureAnalyzer:
    """Analyzer for Kokoro model architecture."""

    def __init__(self, repo_id: str = 'hexgrad/Kokoro-82M', output_dir: str = 'voices/analysis'):
        self.repo_id = repo_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def load_model(self):
        """Load the Kokoro model."""
        print(f"Loading model from {self.repo_id}...")
        self.model = KModel(repo_id=self.repo_id)
        print("Model loaded successfully!")
        return self.model

    def count_parameters(self, module: nn.Module) -> Tuple[int, int]:
        """
        Count total and trainable parameters in a module.

        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def analyze_module_structure(self) -> Dict:
        """Analyze the structure of each major module."""
        if self.model is None:
            self.load_model()

        modules = {
            'bert': self.model.bert,
            'bert_encoder': self.model.bert_encoder,
            'predictor': self.model.predictor,
            'text_encoder': self.model.text_encoder,
            'decoder': self.model.decoder,
        }

        analysis = {}

        for name, module in modules.items():
            total, trainable = self.count_parameters(module)
            analysis[name] = {
                'total_parameters': total,
                'trainable_parameters': trainable,
                'submodules': self._get_submodule_info(module),
            }

        return analysis

    def _get_submodule_info(self, module: nn.Module, max_depth: int = 2) -> Dict:
        """Recursively get information about submodules."""
        info = {}

        for name, submodule in module.named_children():
            total, trainable = self.count_parameters(submodule)
            info[name] = {
                'type': type(submodule).__name__,
                'total_parameters': total,
                'trainable_parameters': trainable,
            }

            # Add shape information for common layer types
            if isinstance(submodule, nn.Linear):
                info[name]['shape'] = f"{submodule.in_features} -> {submodule.out_features}"
            elif isinstance(submodule, nn.Conv1d):
                info[name]['shape'] = f"in={submodule.in_channels}, out={submodule.out_channels}, k={submodule.kernel_size}"
            elif isinstance(submodule, nn.LSTM):
                info[name]['shape'] = f"in={submodule.input_size}, hidden={submodule.hidden_size}, layers={submodule.num_layers}"

        return info

    def analyze_bert_encoder(self) -> Dict:
        """Detailed analysis of BERT encoder."""
        bert = self.model.bert
        config = bert.config

        analysis = {
            'config': {
                'vocab_size': config.vocab_size,
                'embedding_size': config.embedding_size,
                'hidden_size': config.hidden_size,
                'num_hidden_layers': config.num_hidden_layers,
                'num_attention_heads': config.num_attention_heads,
                'intermediate_size': config.intermediate_size,
                'max_position_embeddings': config.max_position_embeddings,
            },
            'layer_structure': [],
        }

        # Analyze each transformer layer
        for i, layer in enumerate(bert.encoder.albert_layer_groups[0].albert_layers):
            layer_info = {
                'layer_id': i,
                'attention': {
                    'num_heads': config.num_attention_heads,
                    'head_dim': config.hidden_size // config.num_attention_heads,
                },
            }
            total, _ = self.count_parameters(layer)
            layer_info['parameters'] = total
            analysis['layer_structure'].append(layer_info)

        return analysis

    def analyze_prosody_predictor(self) -> Dict:
        """Detailed analysis of prosody predictor."""
        predictor = self.model.predictor

        analysis = {
            'text_encoder': {},
            'lstm': {},
            'duration_proj': {},
            'F0_branch': [],
            'N_branch': [],
        }

        # Text encoder (DurationEncoder)
        total, _ = self.count_parameters(predictor.text_encoder)
        analysis['text_encoder'] = {
            'total_parameters': total,
            'num_lstm_layers': len([m for m in predictor.text_encoder.lstms if isinstance(m, nn.LSTM)]),
        }

        # Main LSTM
        lstm = predictor.lstm
        analysis['lstm'] = {
            'input_size': lstm.input_size,
            'hidden_size': lstm.hidden_size,
            'num_layers': lstm.num_layers,
            'bidirectional': lstm.bidirectional,
            'parameters': self.count_parameters(lstm)[0],
        }

        # F0 and N branches
        for i, block in enumerate(predictor.F0):
            total, _ = self.count_parameters(block)
            analysis['F0_branch'].append({
                'block_id': i,
                'type': type(block).__name__,
                'parameters': total,
            })

        for i, block in enumerate(predictor.N):
            total, _ = self.count_parameters(block)
            analysis['N_branch'].append({
                'block_id': i,
                'type': type(block).__name__,
                'parameters': total,
            })

        return analysis

    def analyze_text_encoder(self) -> Dict:
        """Detailed analysis of text encoder."""
        text_encoder = self.model.text_encoder

        analysis = {
            'embedding': {
                'num_embeddings': text_encoder.embedding.num_embeddings,
                'embedding_dim': text_encoder.embedding.embedding_dim,
                'parameters': self.count_parameters(text_encoder.embedding)[0],
            },
            'cnn_layers': [],
            'lstm': {},
        }

        # CNN layers
        for i, cnn_block in enumerate(text_encoder.cnn):
            total, _ = self.count_parameters(cnn_block)
            analysis['cnn_layers'].append({
                'layer_id': i,
                'parameters': total,
            })

        # LSTM
        lstm = text_encoder.lstm
        analysis['lstm'] = {
            'input_size': lstm.input_size,
            'hidden_size': lstm.hidden_size,
            'num_layers': lstm.num_layers,
            'bidirectional': lstm.bidirectional,
            'parameters': self.count_parameters(lstm)[0],
        }

        return analysis

    def analyze_decoder(self) -> Dict:
        """Detailed analysis of decoder (iSTFTNet)."""
        decoder = self.model.decoder

        analysis = {
            'encode': {
                'type': type(decoder.encode).__name__,
                'parameters': self.count_parameters(decoder.encode)[0],
            },
            'decode_blocks': [],
            'generator': {},
        }

        # Decode blocks
        for i, block in enumerate(decoder.decode):
            total, _ = self.count_parameters(block)
            analysis['decode_blocks'].append({
                'block_id': i,
                'type': type(block).__name__,
                'parameters': total,
            })

        # Generator
        generator = decoder.generator
        gen_analysis = {
            'total_parameters': self.count_parameters(generator)[0],
            'num_upsamples': generator.num_upsamples,
            'num_kernels': generator.num_kernels,
            'upsampling_blocks': [],
            'resblocks': [],
        }

        for i, up in enumerate(generator.ups):
            total, _ = self.count_parameters(up)
            gen_analysis['upsampling_blocks'].append({
                'block_id': i,
                'parameters': total,
            })

        for i, res in enumerate(generator.resblocks):
            total, _ = self.count_parameters(res)
            gen_analysis['resblocks'].append({
                'block_id': i,
                'parameters': total,
            })

        analysis['generator'] = gen_analysis

        return analysis

    def analyze_data_flow(self) -> Dict:
        """Analyze the data flow through the model."""
        flow = {
            'input': {
                'phonemes': 'String of phonemes',
                'voice_embedding': 'Tensor [T, 256] where T = len(phonemes)',
            },
            'stages': [
                {
                    'name': 'Input Processing',
                    'description': 'Convert phonemes to input_ids using vocab',
                    'input_shape': '[B, T]',
                    'output_shape': '[B, T]',
                },
                {
                    'name': 'BERT Encoding',
                    'module': 'bert',
                    'description': 'Process input_ids through ALBERT to get contextualized embeddings',
                    'input_shape': '[B, T]',
                    'output_shape': f'[B, T, {self.model.bert.config.hidden_size}]',
                },
                {
                    'name': 'BERT Projection',
                    'module': 'bert_encoder',
                    'description': 'Project BERT outputs to hidden dimension',
                    'input_shape': f'[B, T, {self.model.bert.config.hidden_size}]',
                    'output_shape': '[B, hidden_dim, T]',
                },
                {
                    'name': 'Duration Prediction',
                    'module': 'predictor',
                    'description': 'Predict phoneme durations using prosody style',
                    'style_input': 'voice_embedding[:, 128:]',
                    'output': 'duration tensor',
                },
                {
                    'name': 'F0 and N Prediction',
                    'module': 'predictor.F0Ntrain',
                    'description': 'Predict pitch (F0) and noise (N) contours',
                    'output': 'F0_pred, N_pred',
                },
                {
                    'name': 'Text Encoding',
                    'module': 'text_encoder',
                    'description': 'Encode input through CNN+LSTM',
                    'output_shape': '[B, hidden_dim, T]',
                },
                {
                    'name': 'Alignment',
                    'description': 'Apply predicted alignment to expand features',
                    'output': 'asr (aligned features)',
                },
                {
                    'name': 'Decoding',
                    'module': 'decoder',
                    'description': 'Generate audio using iSTFTNet decoder',
                    'style_input': 'voice_embedding[:, :128]',
                    'output_shape': '[audio_samples]',
                },
            ],
            'output': {
                'audio': 'Waveform tensor at 24kHz sample rate',
                'pred_dur': 'Predicted duration for each phoneme',
            },
        }

        return flow

    def generate_architecture_report(self):
        """Generate comprehensive architecture report."""
        if self.model is None:
            self.load_model()

        print("\n" + "="*80)
        print("KOKORO ARCHITECTURE ANALYSIS")
        print("="*80)

        # Overall statistics
        total_params, trainable_params = self.count_parameters(self.model)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Size: ~{total_params / 1e6:.1f}M parameters")

        # Module-wise analysis
        module_analysis = self.analyze_module_structure()

        print("\n" + "-"*80)
        print("MODULE BREAKDOWN")
        print("-"*80)

        for name, info in module_analysis.items():
            params = info['total_parameters']
            pct = (params / total_params) * 100
            print(f"\n{name.upper()}:")
            print(f"  Parameters: {params:,} ({pct:.1f}%)")

        # Detailed analyses
        bert_analysis = self.analyze_bert_encoder()
        predictor_analysis = self.analyze_prosody_predictor()
        text_enc_analysis = self.analyze_text_encoder()
        decoder_analysis = self.analyze_decoder()
        flow_analysis = self.analyze_data_flow()

        # Save all analyses to JSON
        full_analysis = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'modules': module_analysis,
            'bert_encoder': bert_analysis,
            'prosody_predictor': predictor_analysis,
            'text_encoder': text_enc_analysis,
            'decoder': decoder_analysis,
            'data_flow': flow_analysis,
        }

        output_file = self.output_dir / 'architecture_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(full_analysis, f, indent=2)

        print(f"\n\nSaved detailed analysis to: {output_file}")

        # Generate text report
        self._generate_text_report(full_analysis)

    def _generate_text_report(self, analysis: Dict):
        """Generate human-readable text report."""
        report = []
        report.append("="*80)
        report.append("KOKORO-82M ARCHITECTURE REPORT")
        report.append("="*80)
        report.append("")

        # Overview
        report.append("OVERVIEW")
        report.append("-"*80)
        report.append(f"Total Parameters: {analysis['total_parameters']:,}")
        report.append(f"Model Size: ~{analysis['total_parameters'] / 1e6:.1f}M parameters")
        report.append("")

        # Module breakdown
        report.append("MODULE PARAMETER DISTRIBUTION")
        report.append("-"*80)
        total = analysis['total_parameters']
        for name, info in analysis['modules'].items():
            params = info['total_parameters']
            pct = (params / total) * 100
            report.append(f"{name:20s}: {params:12,} ({pct:5.1f}%)")
        report.append("")

        # BERT Encoder
        report.append("BERT ENCODER (CustomAlbert)")
        report.append("-"*80)
        bert = analysis['bert_encoder']['config']
        report.append(f"Vocabulary Size: {bert['vocab_size']}")
        report.append(f"Hidden Size: {bert['hidden_size']}")
        report.append(f"Num Layers: {bert['num_hidden_layers']}")
        report.append(f"Attention Heads: {bert['num_attention_heads']}")
        report.append(f"Max Position Embeddings: {bert['max_position_embeddings']}")
        report.append("")

        # Prosody Predictor
        report.append("PROSODY PREDICTOR")
        report.append("-"*80)
        predictor = analysis['prosody_predictor']
        report.append(f"Text Encoder Parameters: {predictor['text_encoder']['total_parameters']:,}")
        report.append(f"Main LSTM: {predictor['lstm']['input_size']} -> {predictor['lstm']['hidden_size']} x {predictor['lstm']['num_layers']}")
        report.append(f"F0 Branch Blocks: {len(predictor['F0_branch'])}")
        report.append(f"N Branch Blocks: {len(predictor['N_branch'])}")
        report.append("")

        # Text Encoder
        report.append("TEXT ENCODER")
        report.append("-"*80)
        text_enc = analysis['text_encoder']
        report.append(f"Embedding: {text_enc['embedding']['num_embeddings']} tokens, {text_enc['embedding']['embedding_dim']} dims")
        report.append(f"CNN Layers: {len(text_enc['cnn_layers'])}")
        report.append(f"LSTM: {text_enc['lstm']['input_size']} -> {text_enc['lstm']['hidden_size']}")
        report.append("")

        # Decoder
        report.append("DECODER (iSTFTNet)")
        report.append("-"*80)
        decoder = analysis['decoder']
        report.append(f"Encode Block Parameters: {decoder['encode']['parameters']:,}")
        report.append(f"Decode Blocks: {len(decoder['decode_blocks'])}")
        gen = decoder['generator']
        report.append(f"Generator Parameters: {gen['total_parameters']:,}")
        report.append(f"  Upsampling Blocks: {gen['num_upsamples']}")
        report.append(f"  Residual Blocks: {len(gen['resblocks'])}")
        report.append("")

        # Data Flow
        report.append("DATA FLOW")
        report.append("-"*80)
        for stage in analysis['data_flow']['stages']:
            report.append(f"\n{stage['name']}:")
            report.append(f"  {stage['description']}")
            if 'input_shape' in stage:
                report.append(f"  Input: {stage['input_shape']}")
            if 'output_shape' in stage:
                report.append(f"  Output: {stage['output_shape']}")
        report.append("")

        report.append("="*80)

        # Save report
        report_text = "\n".join(report)
        report_file = self.output_dir / 'architecture_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)

        print(f"\n{report_text}")
        print(f"\nSaved report to: {report_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Kokoro architecture')
    parser.add_argument('--repo-id', default='hexgrad/Kokoro-82M',
                        help='HuggingFace repository ID')
    parser.add_argument('--output-dir', default='voices/analysis',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    analyzer = ArchitectureAnalyzer(args.repo_id, args.output_dir)
    analyzer.generate_architecture_report()


if __name__ == '__main__':
    main()
