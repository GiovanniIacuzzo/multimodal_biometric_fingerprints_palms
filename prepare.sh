# ============================================================
# Setup Script for macOS / Linux
# ============================================================

echo "Setting up Conda environment for Multimodal Biometric..."

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment from environment.yml
if [ -f "config/environment.yml" ]; then
    echo "Creating environment from environment.yml..."
    conda env create -f config/environment.yml
else
    echo "environment.yml not found!"
    exit 1
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate multimodal_biometric_transformer

echo "Environment setup complete!"
echo "To activate later: conda activate multimodal_biometric_transformer"
