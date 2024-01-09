# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# https://github.com/nauyan/Fingers-Dataset-Classification

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import os, glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
EPOCH = 20

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"GPU: {gpus}")
gpu_id = 0
if gpus:
    # Restrict TensorFlow to only use only one GPU based on gpu_id
    try:
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, glob
# print(os.listdir("../input/fingers/fingers/train"))

ROOT_DIR = "/home/zhangjw/projects/ryanz/harker/NeuralNetsHK"
train_img_list = glob.glob(f"{ROOT_DIR}/Images/kaggle-finger-count/train/*.png")
# print(len(train_img_list),
#     len(test_img_list), sep = '\n')
# img = Image.open("../input/fingers/fingers/train/b25805c1-572e-4a9d-ab00-8e4a43a96654_0.png")
# img = np.array(img)
# img = np.reshape(img, (128, 128, -1))
# print(img.shape)
# img_read = io.imread("../input/fingers/fingers/train/b25805c1-572e-4a9d-ab00-8e4a43a96654_0.png")
X_Train = []
Y_Train = []

import keras.utils as np_utils

NUM_INDEX = -6

for img in train_img_list:
    # print(img)
    # label = np_utils.to_categorical(img[-5], 6)
    Y_Train.append(img[NUM_INDEX])
    img = Image.open(img)
    img = np.array(img)
    # print(img.shape)
    img = np.reshape(img, (128, 128, -1))
    # print(img.shape)
    # img_read = transform.resize(img_read, (128,128), mode = 'constant')
    X_Train.append(img)

print("Loading Training Data Done")


X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)
print("Training Data Shape ", X_Train.shape)


Y_Train = np_utils.to_categorical(Y_Train, 6)


from sklearn.model_selection import train_test_split

X_Train, X_Validation, Y_Train, Y_Validation = train_test_split(
    X_Train, Y_Train, test_size=0.2, random_state=1
)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization

model = Sequential()
model.add(Convolution2D(256, (3, 3), padding="same", input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.10))

model.add(Convolution2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.20))

model.add(Convolution2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.30))

model.add(Convolution2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.40))

model.add(Flatten())  # No dropout after flattening.
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(6))
model.add(BatchNormalization())
model.add(Activation("softmax"))

from keras.optimizers import SGD, RMSprop, Adam

opt = SGD(learning_rate=0.01)
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

model.fit(
    X_Train,
    Y_Train,
    batch_size=32,
    epochs=EPOCH,
    verbose=1,
    shuffle=True,
    validation_data=(X_Validation, Y_Validation),
)

import datetime

# Get the current timestamp as a string
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Your desired file name (you can replace "my_file" with your own)
file_name = f"finger_model_{timestamp}.keras"
# Save the model
model.save(file_name)
