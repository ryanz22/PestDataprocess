# Ryan research project

## Development Environment setup

[Getting Started with Conda or Poetry for Data Science Projects]
(<https://medium.com/semantixbr/getting-started-with-conda-or-poetry-for-data-science-projects-1b3add43956d#:~:text=Conda%20and%20Poetry%20stand%20out,environment%20management%20for%20any%20language>.)

### How to install CUDA on Ubuntu

[cuda-1.6.1](https://developer.nvidia.com/cuda-11-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)

[cuda-1.7.1](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)

### How to install PyTorch with Poetry

New method can NOT specify cuda?

```
poetry add torch
```

[Install Pytorch](https://pytorch.org/)

Follow above link to pick the suitable pytorch (verion, os, platform)

Use the command to figure out the exact download url

```
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
```

Use this url in pyproject.toml
```
torch = { url = "https://download.pytorch.org/whl/cu116/torch-1.12.1%2Bcu116-cp310-cp310-linux_x86_64.whl" }
```

### How to run

```sh
> poetry run python3 app.py
```

## HuggingFace Transformers

### TrainingArguments

evaluate_during_training => evaluation_strategy

## How to run Jupyter notebook in VSCode

Prepare python runtime by Pipenv or Poetry

In VSCode, select the corresponding Python interpreter.

## Documents

[huggingface transformers](https://huggingface.co/docs/transformers/index)

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
