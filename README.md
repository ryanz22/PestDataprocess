# Ryan Pest project data process

## Development Environment setup

[Getting Started with Conda or Poetry for Data Science Projects]
(<https://medium.com/semantixbr/getting-started-with-conda-or-poetry-for-data-science-projects-1b3add43956d#:~:text=Conda%20and%20Poetry%20stand%20out,environment%20management%20for%20any%20language>.)

## Linux env setup

```sh
sudo apt install libcairo2-dev libsndfile-dev gobject-introspection libgirepository1.0-dev libsox-dev ffmpeg
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

## Spectrogram

[Audio Deep Learning Made Simple - Why Mel Spectrograms perform better]
(https://ketanhdoshi.github.io/Audio-Mel/)

[Spectrograms, MFCCs, and Inversion in Python]
(https://timsainburg.com/python-mel-compression-inversion.html)

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

[RFCX - Plot time-freq bbox on Log Mel spectrogram]
(https://www.kaggle.com/code/lcolumbo/rfcx-plot-time-freq-bbox-on-log-mel-spectrogram)

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

### Audio augmentation

[A Python library for audio data augmentation. Inspired by albumentations. Useful for machine learning](https://github.com/iver56/audiomentations)

[Scaper tutorial](https://scaper.readthedocs.io/en/latest/tutorial.html)

[Scaper git repo](https://github.com/justinsalamon/scaper)

[PyTorch audio data augmentation](https://pytorch.org/tutorials/beginner/audio_data_augmentation_tutorial.html)

[Tensorflow audio data augmentation](https://www.tensorflow.org/io/tutorials/audio)

[A Survey of Data Augmentation for Audio Classification](https://www.sba.org.br/cba2022/wp-content/uploads/artigos_cba2022/paper_5085.pdf)

[Audio Data Augmentation in python](https://medium.com/@keur.plkar/audio-data-augmentation-in-python-a91600613e47)

## Image processing

### Split large image to small images

[Split-image](https://github.com/whiplashoo/split-image)

```sh
poetry run split-image
```

[Split any image with any degree of overlap](https://github.com/Devyanshu/image-split-with-overlap)

### remove image background

[Rembg is a tool to remove images background](https://github.com/danielgatis/rembg)


## Insect info

### gbif.org

[Search ID by URL](https://www.gbif.org/species/1699053)

Search ID by API

```sh
curl https://api.gbif.org/v1/species/1699053
```

### bugguide.net

Bugguide: https://bugguide.net/node/view/151116

## Datasets

### Sound

[xeno-canto - sharing wildlife sounds from around the world]
(https://xeno-canto.org/)

[xeno-canto Grasshopper sound]
(https://xeno-canto.org/explore/taxonomy?fam=Acrididae)

[xeno-canto grasshopper sound sample](https://xeno-canto.org/species/Arcyptera-kheili)

[LOCUST & GRASSHOPPER SOUNDS](https://www.soundboard.com/sb/Locust_Grasshopper_sounds)

[Avosound grasshopper](https://www.avosound.com/en-us/sound-effects/animal/grasshopper/)

## Shell script

[How To Use bash For Loop In One Line](https://www.cyberciti.biz/faq/linux-unix-bash-for-loop-one-line-command/)

[Bash For Loop Examples](https://www.cyberciti.biz/faq/bash-for-loop/)

Sample

```bash
scripts/multi_files.sh -a 'app/snd_tool.py' -t 'single-slice' \
-g '--length 5.0 --offset 0.2 -f' \
-f 'ls data/sound/grasshopper-sound-4/*mono_44k_denoised.wav'
```

[chmod recursively](https://phoenixnap.com/kb/chmod-recursive)

```sh
sudo find Example -type d -exec chmod 755 {} \;

sudo find Example -type f -exec chmod 644 {} \;
```
