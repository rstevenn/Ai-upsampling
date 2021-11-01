from keras.models import Model
import keras.layers as l
import keras.optimizers as o
import keras.backend as K
from keras.applications import VGG19, VGG16
import tensorflow as tf

def comput_loss(x, y):
    real, fake = x, y
    fake_logit = K.sigmoid(fake - K.mean(real))
    real_logit = K.sigmoid(real - K.mean(fake))
    return fake_logit, real_logit

def SubpixelConv2D( name, scale=2):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    :param scale: upsampling scale compared to input_shape. Default=2
    :return:
    """

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return l.Lambda(subpixel, output_shape=subpixel_shape, name=name)

def vgg19_loss():
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    vgg.trainable = False
    vgg.summary()

    vgg.outputs = [
        vgg.layers[18].output
    ]

    vgg = Model(inputs=vgg.inputs, outputs=vgg.outputs)

    inp = l.Input([128, 128, 3])
    out = l.Lambda(lambda x: x*255)(inp)
    out = vgg(out)
    vgg = Model(inputs=inp, outputs=out)

    return vgg

def get_opti(lr):
    return o.Adam(lr, .5, .99)

def upscaler(input_shape = (None, None, 3), fe_blocks = 6):

    def block_model():
        inp = l.Input((None, None, 64))
        a_path = l.Lambda(lambda x: x[:, :, :, :32])(inp)
        b_path = l.Lambda(lambda x: x[:, :, :, 32:])(inp)

        # A block
        a_ = l.Conv2D(64, kernel_size = 3, padding = 'same')(a_path)
        a = l.Lambda(lambda x: x[:, :, :, :32])(a_)
        b = l.Lambda(lambda x: x[:, :, :, 32:])(a_)

        a = l.Activation("tanh")(a)
        b = l.Activation("sigmoid")(b)
        a_out = l.multiply([a, b])

        # B block
        b = l.Conv2D(64, kernel_size = 3, padding = 'same')(b_path)
        a_ = l.Conv2D(64, kernel_size = 1, padding = 'same')(a_)
        b = l.add([b, a_])

        a = l.Lambda(lambda x: x[:, :, :, :32])(b)
        b = l.Lambda(lambda x: x[:, :, :, 32:])(b)

        a = l.Activation("tanh")(a)
        b = l.Activation("sigmoid")(b)
        b = l.multiply([a, b])

        b = l.Conv2D(32, kernel_size = 1, padding = 'same')(b)
        #b = l.Dropout(.1)(b)
        b_out = l.add([b, b_path])

        out = l.concatenate([a_out, b_out]) 


        return Model(inp, out)

    sub_mod = []
    # input layer
    inp = l.Input(shape=input_shape)
    
    b_out = l.Conv2D(filters = 64, kernel_size = 1, padding = "same")(inp)
    upscaled_recurent = b_out

    sub_mod.append(Model(inp, [b_out, upscaled_recurent]))

    inp0 = l.Input(shape=input_shape)
    b_out, upscaled_recurent = sub_mod[0](inp0)

    # lr blocks
    for _ in range(7):
        last = b_out

        sub_mod.append(block_model())
        b_out = sub_mod[-1](last)

    # up block
    inp = l.Input(shape=[None, None, 64])
    a_out = l.Conv2DTranspose(filters = 64, kernel_size = 3, padding = "same", strides=(2))(inp)
    a_out = l.Activation('relu')(a_out)   
    a_out = l.Conv2DTranspose(filters = 64, kernel_size = 3, padding = "same", strides=(2))(a_out)
    
    b_out = l.UpSampling2D(4, interpolation="bilinear")(inp)
    b_out = l.add([a_out, b_out])

    sub_mod.append(Model(inp, b_out))
    b_out = sub_mod[-1](last)


    # hr blocks
    for _ in range(4):
        last = b_out
        sub_mod.append(block_model())
        b_out = sub_mod[-1](last)

    
    # out block
    inpa = l.Input([None, None, 3])
    inpb = l.Input([None, None, 64])

    outa = l.UpSampling2D(4, interpolation="bilinear")(inpa)
    outb = l.Conv2D(filters = 3, kernel_size = 1, padding = "same")(inpb)
    rgb_out = l.add([outa, outb])

    sub_mod.append(Model([inpa, inpb], rgb_out))
    rgb_out = sub_mod[-1]([inp0, b_out])

    # compile
    model = Model(inp0, rgb_out)
    model.summary(line_length=100)


    # the disc
    inp = l.Input(shape=(128, 128, 3))
    b_out = l.Conv2D(filters = 64, kernel_size = 3, padding = "same")(inp)
    b_out = l.LeakyReLU(.2)(b_out)

    # the blocks
    for i in range(4):
        b_out = l.Conv2D(filters = 64*2**i, kernel_size = 3, padding = "same")(b_out)
        b_out = l.BatchNormalization()(b_out)
        b_out = l.LeakyReLU(.2)(b_out)        

        b_out = l.Conv2D(filters = 64*2**i, kernel_size = 3, padding = "same", strides=2)(b_out)
        b_out = l.BatchNormalization()(b_out)
        b_out = l.LeakyReLU(.2)(b_out) 

    # fusion the patches
    b_out = l.Flatten()(b_out)
    out = l.Dense(1024)(b_out)
    out = l.LeakyReLU(.2)(out)    
    out = l.Dropout(.1)(out)
    out = l.Dense(1)(out)
    #out = l.Lambda(K.mean)(out)

    # compile
    disc = Model(inp, out)
    disc.summary()
    disc.trainable = False

    inp = l.Input(shape=(None, None, 3))
    img = l.UpSampling2D(size=(4, 4), interpolation="bilinear")(inp)
    gan = Model(inp, img)
    gan.summary(line_length=100)


    return model, disc, gan

if __name__ == '__main__':
    import numpy as np
    a , *_ = upscaler((64, 64, 3))
    a.predict(np.random.random([1, 64, 64, 3]))