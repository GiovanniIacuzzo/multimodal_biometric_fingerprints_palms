@echo off
:: ============================================================
:: Setup Script for Windows
:: ============================================================

echo Setting up Conda environment for Multimodal Biometric Transformer...

:: Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda not found. Please install Anaconda or Miniconda first.
    exit /b 1
)

:: Create environment
if exist ./config/environment.yml (
    echo Creating environment from environment.yml...
    conda env create -f ./config/environment.yml
) else (
    echo environment.yml not found!
    exit /b 1
)

:: Activate environment
call conda activate multimodal_biometric_transformer

:: Create folders
:: mkdir data\raw\fingerprints
:: mkdir data\raw\palms
:: mkdir data\processed\train
:: mkdir data\processed\val
:: mkdir data\processed\test
:: mkdir checkpoints
:: mkdir logs

echo Environment setup complete!
echo To activate later, run: conda activate multimodal_biometric_transformer
