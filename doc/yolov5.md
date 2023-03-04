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

## Deep learning

### About transfer learning

https://medium.com/nerd-for-tech/transfer-learning-7914c6ab2b56

https://medium.com/georgian-impact-blog/transfer-learning-part-1-ed0c174ad6e7

https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a


https://towardsdatascience.com/cifar-100-transfer-learning-using-efficientnet-ed3ed7b89af2

**Transfer Learning**

As stated in the Handbook of Research on Machine Learning Applications, transfer learning is the improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned.

In simple terms, transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task. Deep learning networks are resource hungry and computationally expensive with millions of parameters. These networks are trained with a massive amount of data to avoid overfitting. Thus, when a state-of-the-art model is created it often takes researchers a lot of time in training. As a state-of-the-art model is trained after spending such a huge amount of resources, researchers thought that the benefits of such investments should be reaped many times and thus aroused the concept of transfer learning.

## Single label classification for grasshopper sound identification

### dataset

In a folder consists of **train**, **val** and **test** folders.

Each class as the subfolder name and the subfolder has the all image files.

a grasshopper dataset prepared by myself consisting of 16 species soundtracks and 1665 scalogram plots after sound augmentations

### Model

deep learning model: efficientnet_v2_m

https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_m.html#torchvision.models.efficientnet_v2_m

Model summary: 1022 layers, 52,878,852 parameters

EfficientNetV2: Smaller Models and Faster Training

EfficientNets are currently one of the most powerful convolutional neural network (CNN) models. With the rise of Vision Transformers, which achieved even higher accuracies than EfficientNets, the question arose whether CNNs are now dying. EfficientNetV2 proves this wrong by not just improving accuracies but by also reducing training time and latency.

https://towardsdatascience.com/efficientnetv2-faster-smaller-and-higher-accuracy-than-vision-transformers-98e23587bf04

https://arxiv.org/abs/2104.00298

https://paperswithcode.com/method/efficientnetv2

the model is pretrained with ImageNet dataset, acc@1 (on ImageNet-1K): 85.112

with a grasshopper dataset prepared by myself consisting of 16 species soundtracks and 1665 scalogram plots after sound augmentations, acc@1: 98.333

### Train

Initial training with grasshopper dataset based on a pretrained efficientnet_v2_m with ImageNet dataset

```shell
python classify/train.py --model efficientnet_v2_m \
--data ~/work/github/python/PestDataprocess/data/sound/gh-cwt-plot-ds/ \
--project efficientnet_v2 --name gh-cwt-pretrain-100 --img 224 \
--epoch 20 --batch-size 128 --cache 
```

Train efficient from scratch with grasshopper dataset

```shell
python classify/train.py --model efficientnet_v2_m \
--data ~/work/github/python/PestDataprocess/data/sound/gh-cwt-plot-ds/ \
--project efficientnet_v2 --name gh-cwt-from-scratch-20 --img 224 \
--epoch 20 --batch-size 128 --cache --pretrained False
```

Continues training based on previous best weights

```shell
python classify/train.py --weights efficientnet_v2/gh-cwt-imgnet-pretrain-100/weights/best.pt \
--data ~/work/github/python/PestDataprocess/data/sound/gh-cwt-plot-ds/ \
--project efficientnet_v2 --name gh-cwt-from-scratch-20 --img 224 \
--epoch 20 --batch-size 128 --cache
```

### Predict

Predict test data

```shell
python3 classify/predict.py --project efficientnet_v2 \
--name predict-imgnet-pretrain \
--weights efficientnet_v2/gh-cwt-imgnet-pretrain-100/weights/best.pt \
--source ~/work/github/python/PestDataprocess/data/sound/gh-cwt-plot-ds/test/gh-4/ --img 224
```

Predict field data

```shell
python3 classify/predict.py --project efficientnet_v2 \
--name predict-imgnet-pretrain-drone-with-bird-denoise \
--weights efficientnet_v2/gh-cwt-imgnet-pretrain-100/weights/best.pt \
--source ~/work/github/python/PestDataprocess/data/sound/gh-drone/denoise/bird-gh4-peaks_plot/ --img 224
```

## Object detection for grasshopper drone image processing

### dataset

Pretrained with COCO train2017 dataset, mAP_0.5: 66.9, mAP_0.5:0.95: 48.2

My grasshopper object detection dataset consists of 248 512x512 pixel images with image augments (image level: rotate, shear, grayscale, hue, saturation, brightness, exposure, noise; box level: flip, rotate)

### Model

YOLOv5l 46.5M parameters

https://www.analyticsvidhya.com/blog/2021/12/how-to-use-yolo-v5-object-detection-algorithm-for-custom-object-detection-an-example-use-case/#:~:text=It%20is%20a%20novel%20convolutional,and%20probabilities%20for%20each%20component.

https://towardsai.net/p/computer-vision/yolo-v5%E2%80%8A-%E2%80%8Aexplained-and-demystified

**architecture**

https://github.com/ultralytics/yolov5/issues/280

https://blog.roboflow.com/yolov5-improvements-and-evaluation/

https://machinelearningknowledge.ai/introduction-to-yolov5-object-detection-with-tutorial/

### Train

Initial training with grasshopper drone image dataset based on a pretrained yolov5l with COCO dataset

```shell
python train.py --img 512 --batch 32 --epochs 100 --data gh-bugs.yaml \
--weights yolov5l.pt --freeze 10 --project yolov5l --name gh-bugs
```

Continues training based on previous best weights

```shell
python train.py --weights yolov5-obj/gh-bugs/weights/best.pt \
--data gh-bugs.yaml --project yolov5l --name gh-bugs --img 512 \
--epoch 20 --batch-size 128 --cache
```

### Predict

Predict field data

```shell
python detect.py --weights yolov5-obj/5l-gh-only-100/weights/best.pt --img 512 --source ../datasets/image/grasshopper2/test/images/c32_6gh_jpg.rf.5f08a48995ecd7ca2f6c2b97a2ad42d9.jpg --project yolov5-obj --name detect-1
```

## Source separation

NOTE: the current model can only handle 22050Hz sound track

```shell
python3 ../tools/process_wav.py --model_dir bird_mixit_model_checkpoints/output_sources4 \
--checkpoint bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
--num_sources 4 \
--input ~/work/github/python/PestDataprocess/sound/mix/bird-gh-drone-mix-mono-22050.wav \
--output result/bird-gh-drone-mix.wav
```
