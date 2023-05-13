# datasets for insect sound separation

## Core thoughts

two grasshoper sound 2 sec 44.1KHz
one bird sound 2 sec 44.1KHz
drone sound 2 sec 44.1KHz

try mix3 clean where drone as the 3rd src or mix2 noise where drone as noise.

## Datasets

### raw

contains the original sound tracks. Use slice and peaks to generate 'raw-ds'.

### raw-ds

contains the 2 second sound tracks.

### aug-ds

Augment sound tracks with aug tool to make more sound tracks.

### mix-ds

mix strategy 1 - mix2 no noise, gh + bird: loop gh-18 (129 samples),
random pick from bird (68 samples)

CSV header
ID, mix_wav, s1_wav, s2_wav

mix strategy 2 - mix2 + noise, gh + bird + noise drone: loop gh-18 (129 samples),
random pick from bird (68 samples) and drone (36 samples) as noise

CSV header
ID, mix_wav, s1_wav, s2_wav, noise_wav 

mix strategy 3 - mix3 no noise, gh + bird + drone: loop gh-18 (129 samples),
random pick from bird (68 samples) and drone (36 samples)

CSV header
ID, mix_wav, s1_wav, s2_wav, s3_wav

mix strategy 4 - mix3 no noise, gh + gh + drone: loop gh-18 (129 samples),
random pick from gh-21 (82 samples) and drone (36 samples)

CSV header
ID, mix_wav, s1_wav, s2_wav, s3_wav

mix strategy 5 - mix3 + noise, gh + gh + bird + noise drone: loop gh-18 (129 samples),
random pick from gh-21 (82 samples), bird (68 samples) and drone (36 samples) as noise

CSV header
ID, mix_wav, s1_wav, s2_wav, s3_wav, noise_wav

## train-ds

train/val/test

## How to run

To create a mono dataset
```sh
poetry run python app/dataset_tool.py sep-data -i data/sound/insect_sep_ds/raw-2s/ \
-o data/sound/insect_sep_ds/mix2-gh-bird-noise/ --n_src 3 --noise --mux 2
```

To create a train/val/test dataset
```shell
PYTHONPATH=. poetry run python app/dataset_tool.py sep-data \
-i /media/zhangjw/ml-data/projects/ml/datasets/pestdataprocess/sound/insect_sep_ds/raw-2s/ \
-o /media/zhangjw/ml-data/projects/ml/datasets/pestdataprocess/sound/insect_sep_ds/train-mix2-gh-bird-clean/ \
--n_src 2 --train_ds 5 2 1 --fix_len 88200 --main_src gh-18
```
