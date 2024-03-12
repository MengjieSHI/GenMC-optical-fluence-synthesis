import numpy
from numpy import load
from numpy.random import randint
from numpy import ones
from numpy import zeros
from numpy import vstack
from numpy import expand_dims
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from numpy import squeeze
from scipy.io import savemat
from coord import CoordinateChannel2D
import numpy as np


def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    print('data type:', X1.dtype) # didn't perform normalization
    return [X1, X2]


def generate_real_samples(trainA, trainB, n_samples, patch_shape):

    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def summarize_performance(step, g_model, trainA, trainB, n_samples=3):
    # select a sample of input images

    [X_realA, X_realB], _ = generate_real_samples(trainA, trainB, n_samples,1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)



    # plot real source images
    # comment out for only 2-channel source 'images'
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1+i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1+n_samples+i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1+n_samples*2+i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_results/plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_results/model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

def plot_images(src_img, gen_img, tar_img,diff_img):
    images = vstack((src_img, gen_img, tar_img,diff_img))
    # scale from [-1 1] to [0 1]
    # images = (images + 1) / 2.0
    titles = ['Sources', 'Generated', 'Expected','difference']
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, 4, 1+i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # show title
        pyplot.title(titles[i])
    pyplot.show()


=======
import numpy
from numpy import load
from numpy.random import randint
from numpy import ones
from numpy import zeros
from numpy import vstack
from numpy import expand_dims
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from numpy import squeeze
from scipy.io import savemat
from coord import CoordinateChannel2D
import numpy as np


def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    print('data type:', X1.dtype) # didn't perform normalization
    return [X1, X2]


def generate_real_samples(trainA, trainB, n_samples, patch_shape):

    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def summarize_performance(step, g_model, trainA, trainB, n_samples=3):
    # select a sample of input images

    [X_realA, X_realB], _ = generate_real_samples(trainA, trainB, n_samples,1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)



    # plot real source images
    # comment out for only 2-channel source 'images'
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1+i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1+n_samples+i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1+n_samples*2+i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_results/plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_results/model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

def plot_images(src_img, gen_img, tar_img,diff_img):
    images = vstack((src_img, gen_img, tar_img,diff_img))
    # scale from [-1 1] to [0 1]
    # images = (images + 1) / 2.0
    titles = ['Sources', 'Generated', 'Expected','difference']
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, 4, 1+i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # show title
        pyplot.title(titles[i])
    pyplot.show()


>>>>>>> bfb5f61a76607811cd56da41749da16d03d1f831
