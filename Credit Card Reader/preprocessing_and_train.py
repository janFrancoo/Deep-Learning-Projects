import os
import numpy as np
from keras import utils
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

df = np.load(path + "train.npz")
x_train = df['arr_0']
df = np.load(path + "train_label.npz")
y_train = df['arr_0']
df = np.load(path + "test.npz")
x_test = df['arr_0']
df = np.load(path + "test_label.npz")
y_test = df['arr_0']

x_train = x_train.astype("float")
x_test = x_test.astype("float")

x_train /= 255.0
x_test /= 255.0

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

fig, axs = plt.subplots(4, 4)
fig.set_figheight(15)
fig.set_figwidth(15)

for i in range(4):
  for j in range(4):
    randomNum = np.random.randint(0, len(x_train))
    img2copy = x_train[randomNum]
    img = img2copy.copy()
    img[:, :, 0] = img2copy[:, :, 2]
    img[:, :, 2] = img2copy[:, :, 0]
    axs[i, j].imshow(img)
    axs[i, j].set_title(y_train[randomNum])
    
for ax in axs.flat:
    ax.label_outer()

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
 
input_shape = (32, 32, 3)

model = Sequential()
 
model.add(Conv2D(20, (5, 5), padding = "same", input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
 
model.add(Conv2D(50, (5, 5), padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(lr = 0.001), metrics = ['accuracy'])
    
print(model.summary())

from keras.callbacks import ModelCheckpoint
                   
checkpoint = ModelCheckpoint(path + "cc.h5", monitor="val_loss", mode="min", save_best_only = True, verbose=1)

callbacks = [checkpoint]
 
epochs = 5
batchSize = 16

history = model.fit_generator(ImageDataGenerator(rotation_range = 10, width_shift_range = 0.25, 
                    height_shift_range = 0.25, shear_range=0.5, zoom_range=0.5).flow(x_train, y_train, batchSize),
                    steps_per_epoch=len(x_train) / batchSize,
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=(x_test, y_test))
