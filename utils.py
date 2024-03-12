#os pack
import numpy
import h5py
from os import listdir
from numpy import asarray
from numpy import vstack
from numpy import savez_compressed
import tensorflow as tf
# ms: current tf version: 2.0.0; different statement for tf 2.9.0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import load
from matplotlib import pyplot


def load_images(filename, tarName, srcName):

    fData = h5py.File(filename, 'r')
    # read target data (flux)
    tarData = numpy.array(fData.get(tarName))
    print('tar data size:',tarData.shape[0], tarData.shape[1],tarData.shape[2],'type:',tarData.dtype)
    tarData = tarData.reshape(tarData.shape[0], tarData.shape[1], tarData.shape[2], 1)

    srcData = fData.get(srcName)
    print('src data size:', srcData.shape[0], srcData.shape[1], srcData.shape[2], srcData.shape[3], 'type:', srcData.dtype)

    src_array = numpy.zeros((500, 256, 256, 3)) # (number of samples, x_size, y_size, channel)
    src_array[:, :, :, 0] = srcData[:, 0, :, :]
    src_array[:, :, :, 1] = srcData[:, 1, :, :]
    src_array[:, :, :, 2] = srcData[:, 2, :, :]

    srcData = numpy.float32(src_array)

    return [srcData, tarData]


def load_images_4c(filename, tarName, srcName1, srcName2):

    fData = h5py.File(filename, 'r')
    # read target data (flux)
    tarData = numpy.array(fData.get(tarName))
    print('tar data size:',tarData.shape[0], tarData.shape[1],tarData.shape[2],'type:',tarData.dtype)
    tarData = tarData.reshape(tarData.shape[0], tarData.shape[1], tarData.shape[2], 1)

    srcData = fData.get(srcName1)
    srcData_semantic = fData.get(srcName2)
    print('src data size:', srcData.shape[0], srcData.shape[1], srcData.shape[2], 'type:', srcData.dtype)
    # need to modify for different dataset
    src_array = numpy.zeros((400, 128, 128, 4)) # (number of samples, x_size, y_size, channel)

    src_array[:, :, :, 0] = srcData[:, 0, :, :]
    src_array[:, :, :, 1] = srcData[:, 1, :, :]
    src_array[:, :, :, 2] = srcData[:, 2, :, :]
    src_array[:, :, :, 3] = srcData_semantic

    srcData = numpy.float32(src_array)
    return [srcData, tarData]

# dataset path
filename = 'training data/labels_2604_250_500_data_training_256.mat'
# load dataset
[src_images, tar_images] = load_images(filename,'fcw_raw_norm_log_256_filtered','optical_mask_256_filtered')
print('Load:', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'training data/dataset_2604_500_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset:', filename)




# dataset visualisation
data = load('training data/dataset_2604_500_256.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
print('Load:',src_images.shape, tar_images.shape)

# plot source images
n_samples = 3
#src_images = src_images.reshape(src_images.shape[0], src_images.shape[2], src_images.shape[3], src_images.shape[1])
for i in range(n_samples):
    pyplot.subplot(10, n_samples, 1+i)
    pyplot.axis('off')
    src_image=src_images[i]
    #print(src_image.shape)
    pyplot.imshow(src_image[:,:,0]) #

for i in range(n_samples):
    pyplot.subplot(10, n_samples, 1+n_samples+i)
    pyplot.axis('off')
    src_image=src_images[i]
    pyplot.imshow(src_image[:,:,1])

# comment out for 2 channels
for i in range(n_samples):
    pyplot.subplot(10, n_samples, 1+2*n_samples+i)
    pyplot.axis('off')
    src_image=src_images[i]
    #print(src_image.shape)
    pyplot.imshow(src_image[:,:,2])

for i in range(n_samples):
    pyplot.subplot(10, n_samples, 1+3*n_samples+i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i].squeeze())

=======
#os pack
import numpy
import h5py
from os import listdir
from numpy import asarray
from numpy import vstack
from numpy import savez_compressed
import tensorflow as tf
# ms: current tf version: 2.0.0; different statement for tf 2.9.0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import load
from matplotlib import pyplot


def load_images(filename, tarName, srcName):

    fData = h5py.File(filename, 'r')
    # read target data (flux)
    tarData = numpy.array(fData.get(tarName))
    print('tar data size:',tarData.shape[0], tarData.shape[1],tarData.shape[2],'type:',tarData.dtype)
    tarData = tarData.reshape(tarData.shape[0], tarData.shape[1], tarData.shape[2], 1)

    srcData = fData.get(srcName)
    print('src data size:', srcData.shape[0], srcData.shape[1], srcData.shape[2], srcData.shape[3], 'type:', srcData.dtype)

    src_array = numpy.zeros((500, 256, 256, 3)) # (number of samples, x_size, y_size, channel)
    src_array[:, :, :, 0] = srcData[:, 0, :, :]
    src_array[:, :, :, 1] = srcData[:, 1, :, :]
    src_array[:, :, :, 2] = srcData[:, 2, :, :]

    srcData = numpy.float32(src_array)

    return [srcData, tarData]


def load_images_4c(filename, tarName, srcName1, srcName2):

    fData = h5py.File(filename, 'r')
    # read target data (flux)
    tarData = numpy.array(fData.get(tarName))
    print('tar data size:',tarData.shape[0], tarData.shape[1],tarData.shape[2],'type:',tarData.dtype)
    tarData = tarData.reshape(tarData.shape[0], tarData.shape[1], tarData.shape[2], 1)

    srcData = fData.get(srcName1)
    srcData_semantic = fData.get(srcName2)
    print('src data size:', srcData.shape[0], srcData.shape[1], srcData.shape[2], 'type:', srcData.dtype)
    # need to modify for different dataset
    src_array = numpy.zeros((400, 128, 128, 4)) # (number of samples, x_size, y_size, channel)

    src_array[:, :, :, 0] = srcData[:, 0, :, :]
    src_array[:, :, :, 1] = srcData[:, 1, :, :]
    src_array[:, :, :, 2] = srcData[:, 2, :, :]
    src_array[:, :, :, 3] = srcData_semantic

    srcData = numpy.float32(src_array)
    return [srcData, tarData]

# dataset path
filename = 'training data/labels_2604_250_500_data_training_256.mat'
# load dataset
[src_images, tar_images] = load_images(filename,'fcw_raw_norm_log_256_filtered','optical_mask_256_filtered')
print('Load:', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'training data/dataset_2604_500_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset:', filename)




# dataset visualisation
data = load('training data/dataset_2604_500_256.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
print('Load:',src_images.shape, tar_images.shape)

# plot source images
n_samples = 3
#src_images = src_images.reshape(src_images.shape[0], src_images.shape[2], src_images.shape[3], src_images.shape[1])
for i in range(n_samples):
    pyplot.subplot(10, n_samples, 1+i)
    pyplot.axis('off')
    src_image=src_images[i]
    #print(src_image.shape)
    pyplot.imshow(src_image[:,:,0]) #

for i in range(n_samples):
    pyplot.subplot(10, n_samples, 1+n_samples+i)
    pyplot.axis('off')
    src_image=src_images[i]
    pyplot.imshow(src_image[:,:,1])

# comment out for 2 channels
for i in range(n_samples):
    pyplot.subplot(10, n_samples, 1+2*n_samples+i)
    pyplot.axis('off')
    src_image=src_images[i]
    #print(src_image.shape)
    pyplot.imshow(src_image[:,:,2])

for i in range(n_samples):
    pyplot.subplot(10, n_samples, 1+3*n_samples+i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i].squeeze())

>>>>>>> bfb5f61a76607811cd56da41749da16d03d1f831
pyplot.show()