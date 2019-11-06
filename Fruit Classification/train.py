from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

epochs = 10
batchSize = 32
inputShape = (50, 50, 3)

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(71))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

checkpoint = ModelCheckpoint(path + "fruit.h5", monitor="val_loss", mode="min", save_best_only = True, verbose=1)

callbacks = [checkpoint]

history = model.fit_generator(ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True).flow(x_train, y_train, batchSize),
                    steps_per_epoch=len(x_train) / batchSize,
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=(x_test, y_test))
