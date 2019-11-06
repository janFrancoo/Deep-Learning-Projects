import os
import numpy as np
from keras import utils
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from google.colab.patches import cv2_imshow

path = "drive/ColabNotebooks/Fruit Classification/"

df = np.load(path + "train.npz")
x_train = df['arr_0']
df = np.load(path + "train_label.npz")
y_train = df['arr_0']
df = np.load(path + "test.npz")
x_test = df['arr_0']
df = np.load(path + "test_label.npz")
y_test = df['arr_0']

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255.0
x_test /= 255.0

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

labels = ["Apple Braeburn ", "Apple Golden ", "Apple Granny Smith ", "Apple Red ", "Apple Red Delicious ", "Apple Red Yellow ", "Apricot ", "Avocado ",
         "Avocado ripe ", "Banana ", "Banana Red ", "Cactus fruit ", "Cantaloupe ", "Carambula ", "Cherry ", "Cherry Rainier ", "Cherry Wax Black ",
         "Cherry Wax Red ", "Cherry Wax Yellow ", "Clementine ", "Cocos ", "Dates ", "Granadilla ", "Grape Pink ", "Grape White ", "Grapefruit Pink ",
         "Grapefruit White ", "Guava ", "Huckleberry ", "Kaki ", "Kiwi ", "Kumquats ", "Lemon ", "Lemon Meyer ", "Limes ", "Lychee ", "Mandarine ", 
         "Mango ", "Maracuja ", "Melon Piel de Sapo ", "Mulberry ", "Nectarine ", "Orange ", "Papaya ", "Passion Fruit ", "Peach ", "Peach Flat ",
         "Pear ", "Pear Abate ", "Pear Monster ", "Pear Williams ", "Pepino ", "Physalis ", "Physalis with Husk ", "Pineapple ", "Pineapple Mini ",
         "Pitahaya Red ", "Plum ", "Pomegranate ", "Quince ", "Rambutan ", "Raspberry ", "Salak ", "Strawberry ", "Strawberry Wedge ", "Tamarillo ",
         "Tangelo ", "Tomato ", "Tomato Cherry Red ", "Tomato Maroon ", "Walnut "]

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
    axs[i, j].set_title(labels[int(y_train[randomNum])])
    
for ax in axs.flat:
    ax.label_outer()
    
y_train = to_categorical(y_train, num_classes=71)
y_test = to_categorical(y_test, num_classes=71)
