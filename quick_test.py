#!/usr/bin/env python3
"""
Quick test to verify Kokoro installation and basic functionality.
"""

import sys
import torch

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import numpy as np
        import matplotlib
        import seaborn
        import sklearn
        import pandas
        from huggingface_hub import hf_hub_download
        import soundfile
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_kokoro_import():
    """Test Kokoro package import."""
    print("\nTesting Kokoro import...")
    try:
        from kokoro import KModel, KPipeline
        print("✓ Kokoro package imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Kokoro import failed: {e}")
        return False

def test_model_loading():
    """Test model loading."""
    print("\nTesting model loading...")
    try:
        from kokoro import KModel
        model = KModel(repo_id='hexgrad/Kokoro-82M')
        print(f"✓ Model loaded successfully")
        print(f"  - Context length: {model.context_length}")
        print(f"  - Device: {model.device}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_voice_loading():
    """Test voice loading."""
    print("\nTesting voice loading...")
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code='a', model=False)  # Quiet pipeline
        voice = pipeline.load_voice('af_heart')
        print(f"✓ Voice loaded successfully")
        print(f"  - Voice shape: {voice.shape}")
        return True
    except Exception as e:
        print(f"✗ Voice loading failed: {e}")
        return False

def main():
    print("="*60)
    print("Kokoro Quick Test")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Kokoro Import", test_kokoro_import),
        ("Model Loading", test_model_loading),
        ("Voice Loading", test_voice_loading),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Kokoro is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
