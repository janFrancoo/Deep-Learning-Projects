import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from create_siamese_pairs import make_pairs
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Lambda

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = np.expand_dims(train_x, axis=-1)
test_x = np.expand_dims(test_x, axis=-1)

pair_train, label_train = make_pairs(train_x, train_y)
pair_test, label_test = make_pairs(test_x, test_y)

# Config
image_shape = (28, 28, 1)
batch_size = 64
epochs = 10

# Siamese network
inputs = Input(image_shape)
x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.3)(x)

pooled_output = GlobalAveragePooling2D()(x)
outputs = Dense(48)(pooled_output)

model = Model(inputs, outputs)
print(model.summary())

image_a = Input(image_shape)
image_b = Input(image_shape)
feature_a = model(image_a)
feature_b = model(image_b)

distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([feature_a, feature_b])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[image_a, image_b], outputs=outputs)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit([pair_train[:, 0], pair_train[:, 1]], label_train[:], validation_data=(
    [pair_test[:, 0], pair_test[:, 1]], label_test[:]), batch_size=batch_size, epochs=epochs)

model.save("model")
