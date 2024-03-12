from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Cropping2D

from matplotlib import pyplot
from dataset_load import *
from coord import CoordinateChannel2D
from spade_norm import *

import tensorflow as tf

def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    pyplot.figure()
    pyplot.plot(d1_hist, label='d_real')
    pyplot.plot(d2_hist, label='d_fake')
    pyplot.legend()
    pyplot.savefig('plot_results/plot_line_plot_dloss.png')
    pyplot.close()

    pyplot.figure()
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    # save plot to file
    pyplot.savefig('plot_results/plot_line_plot_genloss.png')
    pyplot.close()

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # without restricting to the input size
    in_tar_image = Input(shape=(None, None,1))
    in_src_image = Input(shape=(None, None,3))

    # concatenate images channel-wise
    # ms: concatenate at last dimension
    merged = Concatenate()([in_src_image, in_tar_image])
    #merged_coord = CoordinateChannel2D()(merged)
    # C64
    # ms: keyword argument input_shape=None or tube of integer for the first layer
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_tar_image], patch_out)
    # compile model
    opt = Adam(lr = 0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in) # remove padding='same'
    # conditionallly add batch normalization
    if batchnorm:
        # ms: training is False for discrimator?
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    #g = Activation('relu')(g)
    return g

def decoder_block(layer_in, skip_in, n_filters,initial_mask,dropout=False):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    #g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(layer_in)

    g = Conv2D(n_filters, (3,3), strides=(1,1), padding='same',kernel_initializer=init)(layer_in)
    g = UpSampling2D(size=(2,2), interpolation='bilinear')(g)

    # add batch normalization
    #g = BatchNormalization()(g, training=True)

    # add SPADE norm
    norm_inputs = BatchNormalization()(g,training=False) # add parameter-free normalisation to the activations
    # resize the segmap
    _, inputs_height, inputs_width, _ = g.shape
    segmap_resized = tf.image.resize(initial_mask, size=[inputs_height, inputs_width], method='bilinear')
    # first convolution with activation
    actv = Conv2D(filters=128, kernel_size=(3,3), padding='same')(segmap_resized)
    actv = ReLU()(actv)
    # calculate gamma and beta
    beta = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(actv) # number of filters is hardcoded
    gamma = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(actv)
    # outputs
    g = norm_inputs * (1 + gamma) + beta

    # conditionally drop out
    if dropout:
        g = Dropout(0.2)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])

    # relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

def define_generator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    #in_image = Input(shape=(None, None,3))

    # coordConv module 24-10
    in_image_coord = CoordinateChannel2D()(in_image)
    # encoder model
    e1 = define_encoder_block(in_image_coord, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    #e6 = define_encoder_block(e5, 512)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm no leakyrelu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5) # e4->e5
    b = Activation('relu')(b)
    # decoder model
    #d1 = decoder_block(b,e7,512)
    #d2 = decoder_block(b,e6,512)
    d3 = decoder_block(b, e5, 512, initial_mask=in_image)
    d4 = decoder_block(d3, e4, 512, initial_mask=in_image)  #decoder_block(b, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, initial_mask=in_image)
    d6 = decoder_block(d5, e2, 128, initial_mask=in_image)
    d7 = decoder_block(d6, e1, 64, initial_mask=in_image)
    # output
    #g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    g = Conv2D(1,(3,3),strides=(1,1), padding='same', kernel_initializer=init)(d7)
    g = UpSampling2D(size=(2,2))(g)
    out_image = Activation('relu')(g)
    #out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    model.summary()
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, in_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=in_shape)
    #in_src = Input(shape=(None, None,3))

    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy','mse'],optimizer=opt, loss_weights=[1, 100])
    model.summary()

    return model

def train(d_model, g_model, gan_model, dataset, n_epochs=5, n_batch=1):
    # determine the output square shape of the discriminator
    #n_patch = d_model.output_shape[1]
    #n_patch = 8
    n_patch = 16
    #print(n_patch)
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    # ms: why ?
    n_steps = bat_per_epo * n_epochs
    # create summary
    log_dir = 'logs'
    writer = tf.summary.create_file_writer(log_dir)
    # prepare empty list for storing statis each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of fake samples
        [X_realA, X_realB], y_real = generate_real_samples(trainA,trainB, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2= d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # record loss into summary
        writer.set_as_default()
        tf.summary.scalar('d_loss1-real',d_loss1,step=i)
        tf.summary.scalar('d_loss2-fake',d_loss2,step=i)
        tf.summary.scalar('g_loss',g_loss,step=i)
        # record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)

        # summarize model performance
        if (i+1) % bat_per_epo == 0:
            summarize_performance(i, g_model, trainA, trainB)
            plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)


# load image data
dataset = load_real_samples('dataset_2604_500_256_log.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
#define input shape based on the loaded dataset
in_shape = dataset[0].shape[1:] # (n,n,3)
out_shape = dataset[1].shape[1:] # (n,n,1)
print('size of input:',in_shape)
print('size of output',out_shape)
# define the models without restricting the input size
d_model = define_discriminator(in_shape)
g_model = define_generator(in_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, in_shape=in_shape)
# train model
train(d_model, g_model, gan_model, dataset)
=======
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Cropping2D

from matplotlib import pyplot
from dataset_load import *
from coord import CoordinateChannel2D
from spade_norm import *

import tensorflow as tf

def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    pyplot.figure()
    pyplot.plot(d1_hist, label='d_real')
    pyplot.plot(d2_hist, label='d_fake')
    pyplot.legend()
    pyplot.savefig('plot_results/plot_line_plot_dloss.png')
    pyplot.close()

    pyplot.figure()
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    # save plot to file
    pyplot.savefig('plot_results/plot_line_plot_genloss.png')
    pyplot.close()

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # without restricting to the input size
    in_tar_image = Input(shape=(None, None,1))
    in_src_image = Input(shape=(None, None,3))

    # concatenate images channel-wise
    # ms: concatenate at last dimension
    merged = Concatenate()([in_src_image, in_tar_image])
    #merged_coord = CoordinateChannel2D()(merged)
    # C64
    # ms: keyword argument input_shape=None or tube of integer for the first layer
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_tar_image], patch_out)
    # compile model
    opt = Adam(lr = 0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in) # remove padding='same'
    # conditionallly add batch normalization
    if batchnorm:
        # ms: training is False for discrimator?
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    #g = Activation('relu')(g)
    return g

def decoder_block(layer_in, skip_in, n_filters,initial_mask,dropout=False):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    #g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(layer_in)

    g = Conv2D(n_filters, (3,3), strides=(1,1), padding='same',kernel_initializer=init)(layer_in)
    g = UpSampling2D(size=(2,2), interpolation='bilinear')(g)

    # add batch normalization
    #g = BatchNormalization()(g, training=True)

    # add SPADE norm
    norm_inputs = BatchNormalization()(g,training=False) # add parameter-free normalisation to the activations
    # resize the segmap
    _, inputs_height, inputs_width, _ = g.shape
    segmap_resized = tf.image.resize(initial_mask, size=[inputs_height, inputs_width], method='bilinear')
    # first convolution with activation
    actv = Conv2D(filters=128, kernel_size=(3,3), padding='same')(segmap_resized)
    actv = ReLU()(actv)
    # calculate gamma and beta
    beta = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(actv) # number of filters is hardcoded
    gamma = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(actv)
    # outputs
    g = norm_inputs * (1 + gamma) + beta

    # conditionally drop out
    if dropout:
        g = Dropout(0.2)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])

    # relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

def define_generator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    #in_image = Input(shape=(None, None,3))

    # coordConv module 24-10
    in_image_coord = CoordinateChannel2D()(in_image)
    # encoder model
    e1 = define_encoder_block(in_image_coord, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    #e6 = define_encoder_block(e5, 512)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm no leakyrelu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5) # e4->e5
    b = Activation('relu')(b)
    # decoder model
    #d1 = decoder_block(b,e7,512)
    #d2 = decoder_block(b,e6,512)
    d3 = decoder_block(b, e5, 512, initial_mask=in_image)
    d4 = decoder_block(d3, e4, 512, initial_mask=in_image)  #decoder_block(b, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, initial_mask=in_image)
    d6 = decoder_block(d5, e2, 128, initial_mask=in_image)
    d7 = decoder_block(d6, e1, 64, initial_mask=in_image)
    # output
    #g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    g = Conv2D(1,(3,3),strides=(1,1), padding='same', kernel_initializer=init)(d7)
    g = UpSampling2D(size=(2,2))(g)
    out_image = Activation('relu')(g)
    #out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    model.summary()
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, in_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=in_shape)
    #in_src = Input(shape=(None, None,3))

    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy','mse'],optimizer=opt, loss_weights=[1, 100])
    model.summary()

    return model

def train(d_model, g_model, gan_model, dataset, n_epochs=5, n_batch=1):
    # determine the output square shape of the discriminator
    #n_patch = d_model.output_shape[1]
    #n_patch = 8
    n_patch = 16
    #print(n_patch)
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    # ms: why ?
    n_steps = bat_per_epo * n_epochs
    # create summary
    log_dir = 'logs'
    writer = tf.summary.create_file_writer(log_dir)
    # prepare empty list for storing statis each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of fake samples
        [X_realA, X_realB], y_real = generate_real_samples(trainA,trainB, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2= d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # record loss into summary
        writer.set_as_default()
        tf.summary.scalar('d_loss1-real',d_loss1,step=i)
        tf.summary.scalar('d_loss2-fake',d_loss2,step=i)
        tf.summary.scalar('g_loss',g_loss,step=i)
        # record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)

        # summarize model performance
        if (i+1) % bat_per_epo == 0:
            summarize_performance(i, g_model, trainA, trainB)
            plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)


# load image data
dataset = load_real_samples('dataset_2604_500_256_log.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
#define input shape based on the loaded dataset
in_shape = dataset[0].shape[1:] # (n,n,3)
out_shape = dataset[1].shape[1:] # (n,n,1)
print('size of input:',in_shape)
print('size of output',out_shape)
# define the models without restricting the input size
d_model = define_discriminator(in_shape)
g_model = define_generator(in_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, in_shape=in_shape)
# train model
train(d_model, g_model, gan_model, dataset)
>>>>>>> bfb5f61a76607811cd56da41749da16d03d1f831
