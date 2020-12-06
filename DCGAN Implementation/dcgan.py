from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Conv2D, \
    LeakyReLU, Activation, Flatten, Dense, Reshape


class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, input_dim=100, output_dim=512):
        model = Sequential()
        input_shape = (dim, dim, depth)

        model.add(Dense(input_dim=input_dim, units=output_dim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Reshape(input_shape))
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("tanh"))

        return model

    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):
        model = Sequential()
        input_shape = (height, width, depth)

        model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2), input_shape=input_shape))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        return model
