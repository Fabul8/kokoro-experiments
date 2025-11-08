#!/bin/bash
# Setup script for Kokoro TTS development environment
# This script downloads all necessary dependencies, voices, and prepares the analysis environment

set -e  # Exit on error

echo "=============================================="
echo "Kokoro TTS Setup Script"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $python_version"

# Install system dependencies (espeak-ng)
print_info "Checking for espeak-ng..."
if command -v espeak-ng &> /dev/null; then
    print_success "espeak-ng is already installed"
else
    print_info "Installing espeak-ng..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update -qq
        sudo apt-get install -y espeak-ng > /dev/null 2>&1
        print_success "espeak-ng installed"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install espeak-ng
        print_success "espeak-ng installed"
    else
        print_error "Unsupported OS. Please install espeak-ng manually."
        exit 1
    fi
fi

# Install Python dependencies
print_info "Installing Python dependencies..."
python3 -m pip install -e . -q
python3 -m pip install -q \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    huggingface_hub \
    soundfile \
    tqdm \
    jupyter \
    ipython

print_success "Python dependencies installed"

# Create directories
print_info "Creating directory structure..."
mkdir -p voices
mkdir -p voices/analysis
mkdir -p scripts
mkdir -p outputs
mkdir -p outputs/audio
mkdir -p outputs/graphs
print_success "Directories created"

# Download voices
print_info "Downloading all 63 voices from HuggingFace..."
if [ -f "scripts/download_all_voices.py" ]; then
    python3 scripts/download_all_voices.py
    print_success "Voices downloaded"
else
    print_error "Voice download script not found!"
    exit 1
fi

# Download model weights
print_info "Downloading model weights..."
python3 -c "from kokoro import KModel; KModel(repo_id='hexgrad/Kokoro-82M')"
print_success "Model weights downloaded"

# Run architecture analysis
print_info "Running architecture analysis..."
if [ -f "scripts/analyze_architecture.py" ]; then
    python3 scripts/analyze_architecture.py
    print_success "Architecture analysis complete"
else
    print_error "Architecture analysis script not found!"
fi

# Run voice analysis
print_info "Running voice embedding analysis..."
if [ -f "scripts/analyze_voices.py" ]; then
    python3 scripts/analyze_voices.py
    print_success "Voice analysis complete"
else
    print_error "Voice analysis script not found!"
fi

# Create a test script
print_info "Creating test script..."
cat > test_kokoro.py << 'EOF'
#!/usr/bin/env python3
"""Quick test script to verify Kokoro is working."""

from kokoro import KPipeline
import soundfile as sf

print("Testing Kokoro TTS...")

# Initialize pipeline
pipeline = KPipeline(lang_code='a')

# Test text
text = "Hello! This is a test of the Kokoro text to speech system."

# Generate audio
print(f"Generating audio for: '{text}'")
generator = pipeline(text, voice='af_heart')

# Save first chunk
for i, result in enumerate(generator):
    audio = result.audio
    if audio is not None:
        output_file = f'outputs/audio/test_{i}.wav'
        sf.write(output_file, audio, 24000)
        print(f"✓ Saved: {output_file}")

print("\nTest complete! Kokoro is working correctly.")
EOF

chmod +x test_kokoro.py
python3 test_kokoro.py
print_success "Test script executed successfully"

# Summary
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
echo "  voices/          - Voice embeddings (63 files)"
echo "  voices/analysis/ - Analysis results and graphs"
echo "  scripts/         - Analysis and utility scripts"
echo "  outputs/         - Generated outputs"
echo ""
echo "Available scripts:"
echo "  scripts/download_all_voices.py - Download all voices"
echo "  scripts/analyze_voices.py      - Voice embedding analysis"
echo "  scripts/analyze_architecture.py - Architecture analysis"
echo "  test_kokoro.py                 - Quick test script"
echo ""
echo "Analysis results:"
echo "  voices/analysis/architecture_report.txt"
echo "  voices/analysis/analysis_report.txt"
echo "  voices/analysis/*.png (visualization graphs)"
echo ""
print_success "Kokoro is ready for development!"
