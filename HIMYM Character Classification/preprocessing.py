import os
import numpy as np
from keras import utils
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split

path = "drive/ColabNotebooks/HIMYM Character Classification/"

df = np.load(path + "imgs.npz")
imgs = df['arr_0']
df = np.load(path + "labels.npz")
labels = df['arr_0']

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.33, shuffle=True)

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
    
labels = ["Barney", "Robin", "Lilly", "Marshall", "Ted"]

for i in range(len(y_train)):
  for j in range(len(labels)):
    if y_train[i] == labels[j]:
      y_train[i] = j
      break
      
for i in range(len(y_test)):
  for j in range(len(labels)):
    if y_test[i] == labels[j]:
      y_test[i] = j
      break

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

y_train = utils.to_categorical(y_train, 5)
y_test = utils.to_categorical(y_test, 5)
