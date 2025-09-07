# PestDataprocess Documentation Index

## 1. Project Overview

### 1.1 Introduction and Purpose
The PestDataprocess project is a Python-based toolkit for processing audio data related to pest detection and classification, with a particular focus on grasshoppers and other insects. The system provides tools for audio preprocessing, feature extraction, dataset management, and machine learning preparation.

### 1.2 Environment Setup
- [README.md](README.md) - Development environment setup for Linux and Mac, including required system packages and library installations

### 1.3 Running the Project
- [README.md](README.md) - Instructions for running the CLI tools with PYTHONPATH and poetry

### 1.4 Testing
- [README.md](README.md) - Information on how to run tests with pytest
- [tests/](tests/) - Unit tests for various components

## 2. Datasets

### 2.1 Dataset Overview
- [data/README.md](data/README.md) - General information about datasets including drug review dataset
- [data/sound/README.md](data/sound/README.md) - Comprehensive documentation of sound datasets
- [data/sound/insect_sep_ds/README.md](data/sound/insect_sep_ds/README.md) - Documentation for insect sound separation datasets

### 2.2 Sound Datasets
- Background sounds (environmental noise: bird, cars, river, truck, wind)
- Drone test data with grasshopper, bird, and drone noise
- Grasshopper datasets (raw, augmented, train/val/test splits)
- Mix datasets for testing sound separation
- Scaper audio for data augmentation
- Support sounds collected for the project
- Xeno-canto datasets with grasshopper recordings

### 2.3 Insect Separation Datasets
- Raw soundtracks and processed datasets
- Multiple mixing strategies (mix2, mix3 with and without noise)
- Train/validation/test splits
- CSV file formats for dataset management

## 3. Core Processing Modules

### 3.1 Audio Processing
- [dataprocess/sound/preprocess.py](dataprocess/sound/preprocess.py) - Functions for denoising, mono conversion, resampling, and sound file information
- [dataprocess/sound/audio_filter.py](dataprocess/sound/audio_filter.py) - Audio filter class implementation
- [dataprocess/sound/filter_util.py](dataprocess/sound/filter_util.py) - Filter testing utilities
- [dataprocess/sound/audio_augment.py](dataprocess/sound/audio_augment.py) - Audio augmentation functions
- [dataprocess/sound/audio_id_data.py](dataprocess/sound/audio_id_data.py) - Audio identification data preparation
- [dataprocess/sound/sep_data.py](dataprocess/sound/sep_data.py) - Source separation data handling

### 3.2 Wavelet Transforms
- [dataprocess/cwt/cwt2.py](dataprocess/cwt/cwt2.py) - Continuous Wavelet Transform implementation
- [dataprocess/cwt/scalogram.py](dataprocess/cwt/scalogram.py) - Scalogram generation and visualization

### 3.3 Image Processing
- [dataprocess/image/split.py](dataprocess/image/split.py) - Image splitting with overlap
- [dataprocess/image/__init__.py](dataprocess/image/__init__.py) - Image processing module initialization

### 3.4 Utilities
- [dataprocess/util/data_process.py](dataprocess/util/data_process.py) - General data processing utilities
- [dataprocess/util/file.py](dataprocess/util/file.py) - File and path manipulation utilities
- [dataprocess/util/__init__.py](dataprocess/util/__init__.py) - Utility module initialization

## 4. Command-Line Tools

### 4.1 Sound Processing Tool
- [app/snd_tool.py](app/snd_tool.py) - Comprehensive CLI tool for audio processing with commands for:
  - Denoising, normalization, and format conversion
  - Filtering and spectral analysis
  - Audio slicing and stretching
  - Mixing and augmentation
  - SNR and SI-SNR calculation

### 4.2 Dataset Management Tool
- [app/dataset_tool.py](app/dataset_tool.py) - CLI tool for dataset management with commands for:
  - Splitting datasets into train/val/test sets
  - Image resizing and batch processing
  - Audio visualization (spectrograms, scalograms)
  - Xeno-canto dataset processing
  - Source separation dataset creation

### 4.3 Image Processing Tool
- [app/img_tool.py](app/img_tool.py) - CLI tool for image processing operations

### 4.4 Specialized Tools
- [app/file_tool.py](app/file_tool.py) - File processing utilities
- [app/snd2img.py](app/snd2img.py) - Sound to image conversion tool
- [app/ml/fingers_train.py](app/ml/fingers_train.py) - Machine learning training script
- [app/ml/fingers_eval.py](app/ml/fingers_eval.py) - Machine learning evaluation script

## 5. Machine Learning

### 5.1 Audio Identification
- [dataprocess/sound/audio_id_data.py](dataprocess/sound/audio_id_data.py) - Dataset preparation for audio identification tasks

### 5.2 Source Separation
- [dataprocess/sound/sep_data.py](dataprocess/sound/sep_data.py) - Source separation dataset creation

### 5.3 Training and Evaluation
- [app/ml/fingers_train.py](app/ml/fingers_train.py) - Training scripts
- [app/ml/fingers_eval.py](app/ml/fingers_eval.py) - Evaluation scripts

## 6. References and Resources

### 6.1 External Libraries
- Librosa for audio processing
- PyWavelets for wavelet transforms
- OpenCV for image processing
- Click for CLI framework
- NumPy and SciPy for numerical computations

### 6.2 Academic References
- Wavelet theory and applications
- Spectrogram and mel-scale processing
- CNN architectures (LeNet, AlexNet, VGG, GoogLeNet, ResNet)
- Audio data augmentation techniques

### 6.3 Dataset Sources
- Xeno-canto for bird and insect sound recordings
- GBIF for taxonomic information
- BugGuide for insect identification
- Cornell Bird Dataset

### 6.4 Tutorials and Guides
- Wavelet analysis tutorials
- Audio deep learning resources
- Librosa streaming processing
- Matplotlib visualization techniques
- Data augmentation methods
- Image processing workflows

## Cross-References

- Audio processing functions in [dataprocess/sound/](dataprocess/sound/) are exposed through the CLI tool [app/snd_tool.py](app/snd_tool.py)
- Dataset management functions in [dataprocess/util/](dataprocess/util/) are used by [app/dataset_tool.py](app/dataset_tool.py)
- Wavelet transform implementations in [dataprocess/cwt/](dataprocess/cwt/) are used for feature extraction in various tools
- Image processing utilities in [dataprocess/image/](dataprocess/image/) support dataset visualization
- Machine learning components in [app/ml/](app/ml/) utilize the processed datasets