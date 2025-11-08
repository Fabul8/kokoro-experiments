#!/usr/bin/env python3
"""
Download all available Kokoro voices from HuggingFace.
This script downloads all 63 voice files for analysis.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# All available voices from Kokoro-82M
VOICES = {
    # American English Female
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica',
    'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',

    # American English Male
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
    'am_michael', 'am_onyx', 'am_puck', 'am_santa',

    # British English Female
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',

    # British English Male
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',

    # Other languages and voices
    'ef_dora', 'em_alex', 'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta',
    'hm_omega', 'hm_psi', 'if_sara', 'im_nicola', 'jf_alpha', 'jf_gongitsune',
    'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex', 'pm_santa',
    'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi'
}

def download_all_voices(repo_id='hexgrad/Kokoro-82M', output_dir='voices'):
    """
    Download all voice files from the specified repository.

    Args:
        repo_id: HuggingFace repository ID
        output_dir: Local directory to save voices
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(VOICES)} voices from {repo_id}")
    print(f"Saving to: {output_path.absolute()}")

    failed = []
    succeeded = []

    for voice in tqdm(sorted(VOICES), desc="Downloading voices"):
        try:
            filename = f'voices/{voice}.pt'
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(output_path.parent),
                local_dir_use_symlinks=False
            )
            succeeded.append(voice)
            tqdm.write(f"✓ {voice}")
        except Exception as e:
            failed.append(voice)
            tqdm.write(f"✗ {voice}: {e}")

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Succeeded: {len(succeeded)}/{len(VOICES)}")
    print(f"Failed: {len(failed)}/{len(VOICES)}")

    if failed:
        print(f"\nFailed voices:")
        for voice in failed:
            print(f"  - {voice}")

    return succeeded, failed

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download all Kokoro voices')
    parser.add_argument('--repo-id', default='hexgrad/Kokoro-82M',
                        help='HuggingFace repository ID')
    parser.add_argument('--output-dir', default='voices',
                        help='Output directory for voices')

    args = parser.parse_args()

    download_all_voices(args.repo_id, args.output_dir)
