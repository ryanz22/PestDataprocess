# YoloV5 study notes

## Yolov5 doc

[Official doc](https://docs.ultralytics.com/)

[albumentations - Fast image augmentation library and an easy-to-use wrapper around other libraries]
(https://github.com/albumentations-team/albumentations)

## Environment setup

### Download the git repo

[Yolov5 git repo](https://github.com/ultralytics/yolov5)

copy pyproject.toml

### Comet

[Integrate with Ultralytics YOLOv5](https://www.comet.com/docs/v2/integrations/third-party-tools/yolov5/)

[Track, compare, and reproduce your ML experiments with Comet's machine learning platform]
(https://www.comet.com/zhangjw71#projects)

API key: LyvufUkut0GNwJ3MMSOXyoqd1

```shell
pip install comet_ml  # 1. install
export COMET_API_KEY=LyvufUkut0GNwJ3MMSOXyoqd1 # 2. paste API key
export COMET_PROJECT_NAME=coco128
python3 train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt  # 3. train
```

copy comet.config to yolov5 folder as .comet.config

### wandb.ai

API key: c85eec9543b69482acee21d3d533e55328de9431

existing user you can retrieve your key from https://wandb.ai/authorize

## Single label classification

### dataset

In a folder consists of **train** and **test** folders.

Each class as the subfolder name and the subfolder has the all image files.

### Train

```shell
python3 classify/train.py --model yolov5s-cls.pt --data ./grasshopper/data/images/mini-pest --epochs 5 --img 224 --batch 128
```

### Predict

```shell
python3 classify/predict.py --weights runs/train-cls/exp4/weights/best.pt --source grasshopper/data/images/mini-pest/test/grasshopper/jpg_38.jpg
```
