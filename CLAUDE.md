# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python tools for processing audio data related to pest detection and classification, with a focus on grasshoppers and other insects. The system provides tools for audio preprocessing, feature extraction, dataset management, and machine learning preparation.

The project uses Poetry for dependency management and Python 3.11+ as the target Python version.

## Common Commands

### Development Environment Setup
```bash
# Install system dependencies (Linux)
sudo apt install libcairo2-dev libsndfile-dev gobject-introspection libgirepository1.0-dev libsox-dev ffmpeg python3-dev
sudo apt install libheif1 libheif-dev

# Install system dependencies (Mac)
brew install cairo gobject-introspection libsndfile sox
# Additional linking may be required (see README.md)

# Install Python dependencies
poetry install
```

### Running the Applications
```bash
# Run the sound processing tool
PYTHONPATH=. poetry run python app/snd_tool.py --help

# Run the dataset management tool
PYTHONPATH=. poetry run python app/dataset_tool.py --help

# Example: denoise a sound file
PYTHONPATH=. poetry run python app/snd_tool.py denoise -f input.wav

# Example: normalize audio files
PYTHONPATH=. poetry run python app/snd_tool.py normalize -f input_directory --tsr 22050
```

### Testing
```bash
# Run specific test files
PYTHONPATH=. poetry run pytest tests/test_snd_sep.py

# Run all tests
PYTHONPATH=. poetry run pytest
```

### Code Quality
```bash
# Format code with black
poetry run black .

# Run pylint for code analysis
poetry run pylint app/ dataprocess/

# Run mypy for type checking
poetry run mypy app/ dataprocess/
```

## Code Architecture and Structure

### High-Level Architecture
The project is organized into three main components:

1. **dataprocess/** - Core processing libraries
   - **sound/** - Audio processing functions (preprocessing, filtering, augmentation)
   - **cwt/** - Continuous Wavelet Transform implementations for feature extraction
   - **util/** - Utility functions for file handling and data processing
   - **image/** - Image processing utilities

2. **app/** - Command-line interfaces that expose processing functionality
   - **snd_tool.py** - Primary audio processing CLI with commands for denoising, normalization, format conversion, etc.
   - **dataset_tool.py** - Dataset management CLI for splitting datasets, generating visualizations, and dataset preparation
   - **img_tool.py** - Image processing CLI
   - **ml/** - Machine learning training and evaluation scripts

3. **tests/** - Unit tests for the processing functions

### Key Processing Flows

1. **Audio Preprocessing Pipeline**
   - Raw audio files (various formats) are loaded using librosa
   - Preprocessing functions in `dataprocess/sound/preprocess.py` handle denoising, mono conversion, and resampling
   - Audio augmentation is handled by `dataprocess/sound/audio_augment.py`

2. **Feature Extraction**
   - Continuous Wavelet Transform implementations in `dataprocess/cwt/cwt2.py` and `dataprocess/cwt/scalogram.py` 
   - These generate time-frequency representations for machine learning

3. **Dataset Management**
   - Tools in `app/dataset_tool.py` handle dataset splitting, visualization generation, and preparation
   - Source separation dataset creation is handled by `dataprocess/sound/sep_data.py`

### Important Dependencies
- librosa - Audio processing
- PyWavelets - Wavelet transforms
- OpenCV - Image processing
- Click - CLI framework
- NumPy/SciPy - Numerical computing
- noisereduce - Audio denoising
- scaper - Audio data augmentation