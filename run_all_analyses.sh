#!/bin/bash
# Master script to run all Kokoro analyses
# Run this after setup is complete

set -e

echo "=============================================="
echo "Kokoro Complete Analysis Pipeline"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${YELLOW}>>> $1${NC}"
}

print_done() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Check if voices exist
if [ ! -d "voices" ] || [ -z "$(ls -A voices/*.pt 2>/dev/null)" ]; then
    print_step "Downloading voices..."
    python3 scripts/download_all_voices.py
    print_done "Voices downloaded"
else
    echo "✓ Voices already downloaded"
fi

# Architecture analysis
print_step "Running architecture analysis..."
python3 scripts/analyze_architecture.py
print_done "Architecture analysis complete"

# Standard voice analysis
print_step "Running standard voice embedding analysis..."
python3 scripts/analyze_voices.py
print_done "Voice analysis complete"

# Incremental PCA analysis
print_step "Running incremental PCA analysis..."
python3 scripts/incremental_pca_analysis.py
print_done "Incremental PCA analysis complete"

echo ""
echo "=============================================="
echo "All Analyses Complete!"
echo "=============================================="
echo ""
echo "Results saved to: voices/analysis/"
echo ""
echo "Generated files:"
echo "  - architecture_analysis.json"
echo "  - architecture_report.txt"
echo "  - analysis_report.txt"
echo "  - incremental_pca_report.txt"
echo "  - incremental_pca_{decoder,prosody,both}.json"
echo "  - centroids.npz"
echo "  - centroid_stats.json"
echo "  - top_variable_dimensions.json"
echo ""
echo "Generated visualizations:"
echo "  - pca_variance.png"
echo "  - pca_2d.png"
echo "  - tsne_both.png"
echo "  - interlayer_correlation.png"
echo "  - embedding_distributions.png"
echo "  - dimension_variance.png"
echo "  - incremental_pca_{decoder,prosody,both}.png"
echo "  - group_contribution_{decoder,prosody,both}.png"
echo ""
