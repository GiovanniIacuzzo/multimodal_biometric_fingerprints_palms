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
call conda activate multimodal_biometric

echo Environment setup complete!
echo To activate later, run: conda activate multimodal_biometric

echo Creating project directories...


set "Result_DIR=results"

:: Crea la cartella principale (se non esiste)
if not exist "%Result_DIR%" mkdir "%Result_DIR%"

:: Crea le sottocartelle
if not exist "%Result_DIR%\img" mkdir "%Result_DIR%\img"
if not exist "%Result_DIR%\logs" mkdir "%Result_DIR%\logs"

echo Folders created:
echo   - %Result_DIR%\img
echo   - %Result_DIR%\logs

