import scipy
import imageio
import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications import inception_v3

K.set_learning_phase(0)

#model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
model = vgg16.VGG16(weights='imagenet', include_top=False)
dream = model.input

layer_dict = dict([(layer.name, layer) for layer in model.layers])

"""settings = {
    'features': {
        'mixed2': 0.9,
        'mixed3': 1.2,
        'mixed4': 1.8,
        'mixed5': 0.9,
    },
}"""

settings = {
    'features': {
        'block4_conv1': 0.08,
        'block4_conv2': 0.03,
        'block4_conv3': 0.04
    },
}

for layer_name in settings['features']:
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    loss = coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imageio.imwrite(fname, pil_img)


def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


step = 0.01
num_octave = 3
octave_scale = 1.2
iterations = 20
max_loss = 20.0

base_image_path = '1.jpg'

img = preprocess_image(base_image_path)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)

    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

save_img(img, fname='final_dream.png')
print("DeepDreaming Complete")
