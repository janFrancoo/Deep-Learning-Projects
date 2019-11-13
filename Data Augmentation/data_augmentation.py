from keras.datasets import mnist
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
    
pyplot.show()

train_datagen = ImageDataGenerator(rotation_range=60)
train_datagen.fit(x_train)

for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(x_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break

train_datagen = ImageDataGenerator(shear_range=0.5, zoom_range=0.5)
train_datagen.fit(x_train)

for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(x_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break

train_datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True)
train_datagen.fit(x_train)

for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(x_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break

train_datagen = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)
train_datagen.fit(x_train)

for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(x_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break

train_datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_datagen.fit(x_train)

for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(x_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break
