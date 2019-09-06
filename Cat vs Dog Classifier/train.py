import numpy as np

def load_catdog_dataset():
  
  npzFile = np.load("drive/train.npz")
  train = npzFile['arr_0']
  
  npzFile = np.load("drive/train_labels.npz")
  trainLabels = npzFile['arr_0']
  
  npzFile = np.load("drive/test.npz")
  test = npzFile['arr_0']
  
  npzFile = np.load("drive/test_labels.npz")
  testLabels = npzFile['arr_0']
  
  return train, trainLabels, test, testLabels


x_train, y_train, x_test, y_test = load_catdog_dataset()

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from google.colab.patches import cv2_imshow

for i in range(10):
  random = np.random.randint(0, len(x_test))
  img = x_test[random]
  print(y_test[random])
  cv2_imshow(img)
  
  from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

epochs = 25
batchSize = 16
inputShape = (150, 150, 3)

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, batch_size = batchSize, epochs = epochs, validation_data = (x_test, y_test), shuffle = True)
          
model.save("drive/ColabNotebooks/catsvsdogs.h5")

import matplotlib.pyplot as plt

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

import matplotlib.pyplot as plt

history_dict = history.history

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
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
