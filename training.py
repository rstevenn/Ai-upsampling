from model import *
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random as rd
import os
from threading import Thread

import tensorflow as tf

# PATHS
DATA_DIR = "data/"
MODE_DIR = "model/"
IMAGE_DIR = "ims/"

# HYPERPARAMETTERS
BATCH_SIZE = 8
IMG_SIZE = 128
TRAINING_STEP = 100_000

# data generation
def get_batch(image_list, current_image):
    
    X = []
    Y = []

    def get_images(X, Y, image):
        # select and read an image
        image_path = os.path.join(DATA_DIR, image)
        image = Image.open(image_path)
        image = image.convert('RGB')

        # crop the image and resize
        x, y = image.size
        while x < IMG_SIZE+1 or y < IMG_SIZE+1:
            image = image.resize([int(x*2), int(y*2)])
            x, y = image.size

        
        x_base , y_base = rd.randint(0, x-IMG_SIZE), rd.randint(0, y-IMG_SIZE)
        selected_square = image.crop((x_base, y_base, x_base+IMG_SIZE, y_base+IMG_SIZE))
        low_quality_square = selected_square.resize((IMG_SIZE // 4, IMG_SIZE // 4))

        # add to the batch and convert color
        X.append(np.array(low_quality_square, dtype='float32') / 255 + (np.random.normal(0, 0.2, size = np.array(low_quality_square).shape) if rd.random() > .9 else 0))
        Y.append(np.array(selected_square, dtype='float32') / 255 )

    ths = []
    for i in range(BATCH_SIZE):
        th = Thread(target=get_images, args=(X, Y, image_list[current_image]))
        th.start()
        ths.append(th)
        current_image = (current_image + 1)

        if current_image == len(image_list):
            current_image = 0
            rd.shuffle(image_list)

    for th in ths:
        th.join()

    return np.array(X, dtype='float32'), np.array(Y, dtype='float32'), current_image

@tf.function
def train_step(x, y, ones, zeros, disc, gen, mse, disc_opt, gen_opt, mse_opt, vggloss):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as mse_tape:
        # forward
        gen_ = gen(x, training=True)
        print(gen_)
        disc1 = disc(y, training=True)
        disc2 = disc(gen_, training=True)

        # losses
        discb, disca = comput_loss(disc1, disc2)
        dloss1 = K.binary_crossentropy(disca, ones)
        dloss2 = K.binary_crossentropy(discb, zeros)
        dloss =  K.mean(dloss1 + dloss2)

        t, p = vgg(y, training=False), vgg(gen_, training=False)
        gloss1 = K.mean(K.binary_crossentropy(disc2, ones) + K.binary_crossentropy(disc1, zeros)) * .001 + K.mean(K.abs(t[0] - p[0]))

        gen_ = mse(x, training=True)
        t, p = vgg(y, training=False), vgg(gen_, training=False)
        gloss2 = K.mean(K.abs(t[0] - p[0])) + K.mean(K.abs(y - gen_))

        gloss = gloss1 #+ gloss3

    # apply gradient
    gradients_of_generator = gen_tape.gradient(gloss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(dloss, disc.trainable_variables)
    gradients_of_mse = mse_tape.gradient(gloss2, mse.trainable_variables)

    #Apply gradients
    gen_opt.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
    mse_opt.apply_gradients(zip(gradients_of_mse, mse.trainable_variables))

    # return losses
    return [gloss1, gloss2], [K.mean(dloss1), K.mean(dloss2)]

train = train_step

# get usefull data
bi, *_ = upscaler()
mse, *_ = upscaler()
model, disc, gan = upscaler()
disc_opt, gen_opt, mse_opt = get_opti(0.0002), get_opti(0.0001), get_opti(0.0001)

vgg = vgg19_loss()
images_path = os.listdir(DATA_DIR)
rd.shuffle(images_path)

current_image = 0
# the training loop
for i in range(TRAINING_STEP):
    x, y, current_image = get_batch(images_path, current_image)
    ones, zeros = np.ones(shape=[BATCH_SIZE], dtype='float32'), np.zeros(shape=[BATCH_SIZE], dtype='float32')

    gloss, dloss = train(x, y, ones, zeros, disc, model, mse, disc_opt, gen_opt, mse_opt, vgg)


    print(f'step {i + 1} epoch {((i+1)*BATCH_SIZE)//len(images_path)} dloss [{np.array(dloss[0])} {np.array(dloss[1])}] gloss [{np.array(gloss[0])} {np.array(gloss[1])}]')

    if i%50== 0:
        # interpolate the 2 models
        for l in range(len(model.layers)):
            gan_w = model.layers[l].get_weights()
            mse_w = mse.layers[l].get_weights()

            weights = []
            gw = []
            mw = []
            for k in range(len(mse_w)):
                weights.append(gan_w[k] * .5 + mse_w[k] * .5)
                gw.append(gan_w[k] * .9 + mse_w[k] * .1)
                mw.append(gan_w[k] * .1 + mse_w[k] * .9)

            if i%150 == 0:
                model.layers[l].set_weights(mw)
                mse.layers[l].set_weights(gw)
                
            bi.layers[l].set_weights(weights)            


        image_path = os.path.join(DATA_DIR, rd.choice(images_path))
        image = Image.open(image_path)
        image = image.convert('RGB')
        x, y = image.size
        ratio = 256/max([x, y])
        image = image.resize((int(x*ratio), int(y*ratio)))
        y = np.array(image.resize((int(x*ratio)*4, int(y*ratio)*4), Image.NEAREST)) / 255
        x = np.array([np.array(image) / 255], dtype='float32')


        with tf.device('/cpu:0'):
            image_bi = bi.predict(x)[0]
            image_gn = model.predict(x)[0]
            image_ms = mse.predict(x)[0]
            x = gan.predict(x)

        im = np.zeros(shape=[x.shape[1]*2, x.shape[2] * 3, 3])
        im[:x.shape[1], 0:x.shape[2]] = x[0]
        im[:x.shape[1], x.shape[2]: x.shape[2]*2] = image_bi
        im[:x.shape[1], x.shape[2]*2:] = np.abs(y - image_bi)/2
        im[x.shape[1]:, 0:x.shape[2]] = image_gn
        im[x.shape[1]:,x.shape[2]: x.shape[2]*2] = image_gn * .5 + image_ms * .5
        im[x.shape[1]:, x.shape[2]*2:] = image_ms

        im = im.clip(0, 1)
        im *= 255
        im = np.array(im, dtype ='uint8')

        im = Image.fromarray(im)
        im.save(IMAGE_DIR + f"{i / 10 + 1}.png")


        bi.save(MODE_DIR + f"{i / 10 + 1}_bi.h5")
        mse.save(MODE_DIR + f"{i / 10 + 1}_mse.h5")
        model.save(MODE_DIR + f"{i / 10 + 1}_gan.h5")