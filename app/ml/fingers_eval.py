import numpy as np
import keras.utils as np_utils
import os, glob, sys
from PIL import Image
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT_DIR = "/home/zhangjw/projects/ryanz/harker/NeuralNetsHK"
NUM_INDEX = -6

# test_img_list = glob.glob(f"{ROOT_DIR}/Images/kaggle-finger-count/test/*.png")
test_img_list = glob.glob(
    f"{ROOT_DIR}/Images/kaggle-finger-count/small-finger-count/*.png"
)
if len(test_img_list) == 0:
    print("Can NOT find png files")
    sys.exit(1)

X_Test = []
Y_Test = []

for img in test_img_list:
    # print(img)
    # label = np_utils.to_categorical(img[-5], 6)
    Y_Test.append(img[NUM_INDEX])
    img = Image.open(img)
    img = np.array(img)
    # print(img.shape)
    img = np.reshape(img, (128, 128, -1))
    # print(img.shape)
    # img_read = transform.resize(img_read, (128,128), mode = 'constant')
    X_Test.append(img)

print("Loading Test Data Done")

X_Test = np.array(X_Test)
# X_Test /= 255
Y_Test = np.array(Y_Test)
# Y_Test /= 255
print("Test Data Shape ", X_Test.shape)

Y_Test = np_utils.to_categorical(Y_Test, 6)


FN = "finger_model_2024-01-08_17-22-24.keras"
# Load the saved model later if needed
loaded_model = tf.keras.models.load_model(FN)

from sklearn.metrics import classification_report, confusion_matrix

Y_pred = loaded_model.predict_step(X_Test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)


y_pred = loaded_model.predict(X_Test).round()
print(y_pred)

target_names = ["0", "1", "2", "3", "4", "5"]
print(
    classification_report(np.argmax(Y_Test, axis=1), y_pred, target_names=target_names)
)
print(confusion_matrix(np.argmax(Y_Test, axis=1), y_pred))

pred = loaded_model.evaluate(X_Test, Y_Test, batch_size=32)

print("Accuracy of model on test data is: ", pred[1] * 100)
