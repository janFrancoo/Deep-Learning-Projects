import os
import cv2
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from create_dataset import get_data
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import SeparableConv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    print("install CUDA or train on cloud")
    sys.exit()

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="Path of faces")
ap.add_argument("-m", "--model", required=True, help="Out path of model after train")
args = vars(ap.parse_args())

faces, labels = get_data(args["face"])
"""
face_x = 0
face_y = 0
for face in faces:
    face_x += face.shape[1]
    face_y += face.shape[0]

mean_face_x = int(face_x / len(faces))
mean_face_y = int(face_y / len(faces))

input_shape = (mean_face_x, mean_face_x) if mean_face_x < mean_face_y else (mean_face_y, mean_face_y)
"""
for i, face in enumerate(faces):
    faces[i] = cv2.resize(face, (128, 128))

faces = np.array(faces)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(faces, labels, test_size=0.3)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255.0
x_test /= 255.0

epochs = 15
batch_size = 48
input_shape = (128, 128, 3)
chan_dim = -1

model = Sequential()

model.add(SeparableConv2D(16, (7, 7), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chan_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chan_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chan_dim))

model.add(SeparableConv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chan_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2 / 50)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test),
                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs)
model.save(args["model"])

label_names = ['Non-masked', 'Masked']
res = model.predict_classes(x_test, batch_size=64)
print(classification_report(y_test, res, target_names=label_names))

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label="Validation Loss")
line2 = plt.plot(epochs, loss_values, label="Training Loss")
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label="Validation Accuracy")
line2 = plt.plot(epochs, acc_values, label="Training Accuracy")
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()
