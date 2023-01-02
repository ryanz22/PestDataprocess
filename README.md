# Ryan Pest project data process

## Development Environment setup

[Getting Started with Conda or Poetry for Data Science Projects]
(<https://medium.com/semantixbr/getting-started-with-conda-or-poetry-for-data-science-projects-1b3add43956d#:~:text=Conda%20and%20Poetry%20stand%20out,environment%20management%20for%20any%20language>.)

## Linux env setup

```sh
sudo apt install libcairo2-dev libsndfile-dev gobject-introspection libgirepository1.0-dev libsox-dev
```

## Macbook env setup

```sh
brew install cairo gobject-introspection libsndfile
```

### How to run

```sh
PYTHONPATH=. poetry run python3 app/snd_tool.py
```

[Rename all files in directory](https://stackoverflow.com/questions/7450818/rename-all-files-in-directory-from-filename-h-to-filename-half)

rename all files in a folder from x.y to bx.y

```sh
for f in *.jpg; do mv $f "b$f"; done
```

## Wavelet

[A gentle introduction to wavelet for data analysis](https://www.kaggle.com/code/asauve/a-gentle-introduction-to-wavelet-for-data-analysis/notebook)

[ECG Signals Classification using Continuous Wavelet Transform (CWT) and Deep Neural Network](https://www.youtube.com/watch?v=rI6A2lKTM10)

[Continuous Wavelet Transform and Scale-Based Analysis](https://www.mathworks.com/help/wavelet/gs/continuous-wavelet-transform-and-scale-based-analysis.html)

## Librosa

[Why resample on load?](https://librosa.org/blog/2019/07/17/resample-on-load/#resample-on-load)

[Streaming for large files](https://librosa.org/blog/2019/07/29/stream-processing/#stream-processing)

## CNN

[CNN Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet and more]
(https://medium.com/analytics-vidhya/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)

[PyTorch: Directly use pre-trained AlexNet for Image Classification and Visualization of the activation maps]
(https://medium.com/analytics-vidhya/pytorch-directly-use-pre-trained-alexnet-for-image-classification-and-visualization-of-the-dea0de3eade9)

[Transfer Learning using Pre-Trained AlexNet Model and Fashion-MNIST]
(https://towardsdatascience.com/transfer-learning-using-pre-trained-alexnet-model-and-fashion-mnist-43898c2966fb)

## Matplotlib

[Matplotlib, Pyplot, Pylab etc: What's the difference between these and when to use each?]
(https://queirozf.com/entries/matplotlib-pylab-pyplot-etc-what-s-the-different-between-these)

## Kaggle

[How To Download Dataset From Kaggle](https://www.ankushchoubey.com/download_kaggle/)

## Sound processing

Make sound louder

```sh
ffmpeg -i data/sound/grasshopper-sound-4/gh-4_mono_22050_denoised.wav \
  -filter:a 'volume=3.0' \
  data/sound/grasshopper-sound-4/gh-4_mono_22050_denoised_louder.wav
```

### Denoise

[Noise reduction in python using spectral gating](https://github.com/timsainb/noisereduce)

[A wavelet audio denoiser](https://github.com/actondev/wavelet-denoiser)

very good with method=dwt

```sh
~/.cache/pypoetry/virtualenvs/pestdataprocess-pxU8UH9y-py3.10/bin/python3 \
  src/denoiser-argument.py \
  -i ~/work/github/python/PestDataprocess/data/sound/grasshopper-sound-4/gh-4_mono_22050.wav \
  -method=dwt -wavelet dmey2 -o tmp2.wav
```

## Image processing

### Split large image to small images

[Split-image](https://github.com/whiplashoo/split-image)

```sh
poetry run split-image
```

[Split any image with any degree of overlap](https://github.com/Devyanshu/image-split-with-overlap)

## Insect info

### gbif.org

[Search ID by URL](https://www.gbif.org/species/1699053)

Search ID by API

```sh
curl https://api.gbif.org/v1/species/1699053
```

### bugguide.net

Bugguide: https://bugguide.net/node/view/151116
