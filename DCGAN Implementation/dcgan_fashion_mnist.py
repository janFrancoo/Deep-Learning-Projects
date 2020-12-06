import os
import cv2
import argparse
import numpy as np
from dcgan import DCGAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from sklearn.utils import shuffle
from imutils import build_montages

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=50, help="# epochs to train for")
ap.add_argument("-b", "--batch_size", type=int, default=128, help="batch size for training")
args = vars(ap.parse_args())

epochs = args["epochs"]
batch_size = args["batch_size"]
init_lr = 2e-4

((train_x, _), (test_x, _)) = fashion_mnist.load_data()
train_images = np.concatenate([train_x, test_x])
train_images = np.expand_dims(train_images, axis=-1)
train_images = (train_images.astype("float") - 127.5) / 127.5

generator = DCGAN.build_generator(7, 64, channels=1)
discriminator = DCGAN.build_discriminator(28, 28, 1)
discriminator_opt = Adam(lr=init_lr, beta_1=0.5, decay=init_lr / epochs)
discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_opt)

discriminator.trainable = False
gan_input = Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan_opt = Adam(lr=init_lr, beta_1=0.5, decay=init_lr / epochs)
gan.compile(loss="binary_crossentropy", optimizer=gan_opt)

benchmark_noise = np.random.uniform(-1, 1, size=(256, 100))
for epoch in range(0, epochs):
    print("[INFO] starting epoch {} of {}...".format(epoch + 1, epochs))
    batches_per_epoch = int(train_images.shape[0] / batch_size)

    for i in range(0, batches_per_epoch):
        noise = np.random.uniform(-1, 1, size=(batch_size, 100))
        generated_images = generator.predict(noise, verbose=0)

        image_batch = train_images[i * batch_size:(i + 1) * batch_size]
        x = np.concatenate((image_batch, generated_images))
        y = ([1] * batch_size) + ([0] * batch_size)
        y = np.reshape(y, (-1,))
        (x, y) = shuffle(x, y)
        discriminator_loss = discriminator.train_on_batch(x, y)

        noise = np.random.uniform(-1, 1, (batch_size, 100))
        fake_labels = [1] * batch_size
        fake_labels = np.reshape(fake_labels, (-1,))
        gan_loss = gan.train_on_batch(noise, fake_labels)

        p = None
        if i == batches_per_epoch - 1:
            p = [args["output"], "epoch_{}_output.png".format(str(epoch + 1).zfill(4))]
        else:
            if epoch < 10 and i % 25 == 0:
                p = [args["output"], "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]
            elif epoch >= 10 and i % 100 == 0:
                p = [args["output"], "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]

        if p is not None:
            print("[INFO] Step {}_{}: discriminator_loss={:.6f}, "
                  "adversarial_loss={:.6f}".format(epoch + 1, i, discriminator_loss, gan_loss))

            images = generator.predict(benchmark_noise)
            images = ((images * 127.5) + 127.5).astype("uint8")
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (16, 16))[0]

            p = os.path.sep.join(p)
            cv2.imwrite(p, vis)
