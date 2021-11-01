import tensorflow.keras.models as m
import numpy as np

import tensorflow as tf
from keras.layers import Input
from keras.models import Model

from PIL import Image
import os

model = m.load_model("selected/1851.0_gan.h5")
model.summary(line_length=100)

# split the model
def split(model):
    
    models = []

    # the input model
    inp = Input([None, None, 3])
    out = model.layers[1](inp)[0]
    models.append(Model(inp, out))

    # the features blocks
    for i in range(len(model.layers)-3):
        inp = Input([None, None, 64])
        out = model.layers[i+2](inp)
        models.append(Model(inp, out))

    # the last block
    inpa = Input([None, None, 3])
    inpb = Input([None, None, 64])
    out = model.layers[-1]([inpa, inpb])
    models.append(Model([inpa, inpb], out))

    return models

models = split(model)

for name in os.listdir('lr'):
    # load the image
    path = 'lr/' + name
    image = Image.open(path)
    image = image.convert('RGB')
    image = np.array([np.array(image, dtype='float32') / 255])
    
    # predict
    with tf.device('/cpu:0'):
        out = image
        for model in models[:-2]:
            out = model.predict(out)

        out = models[-1].predict([image, out])[0]

    # post processing
    out = out.clip(0, 1)
    out = np.array(out * 255, dtype='uint8')

    # save
    image = Image.fromarray(out)
    image.save('hr/' + name)