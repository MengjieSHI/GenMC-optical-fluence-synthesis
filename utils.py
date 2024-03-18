#os pack
import numpy as np
import h5py
from os import listdir
from numpy import asarray
from numpy import vstack


#import tensorflow as tf
# ms: current tf version: 2.0.0; different statement for tf 2.9.0
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing.image import load_img

from matplotlib import pyplot
from matplotlib import transforms

# convert dataset in .mat format into .npz for model training
def load_mat_data(file_path, tarName, srcName):

    fData = h5py.File(file_path, 'r')
    tarData = np.array(fData.get(tarName))
    tarData = tarData.reshape(tarData.shape[0], tarData.shape[1], tarData.shape[2], 1)
    print('tar data (optical fluence) size: number of samples:%d, per sample dimension:(%d, %d, %d)'% (tarData.shape[0], tarData.shape[1],tarData.shape[2], tarData.shape[3]),'data length:',tarData.dtype)

    srcData = np.array(fData.get(srcName))
    srcData = srcData.reshape(srcData.shape[0], srcData.shape[2], srcData.shape[3], srcData.shape[1])
    print('src data (optical properties) size: number of samples:%d, per sample dimension:(%d, %d, %d)'% (srcData.shape[0], srcData.shape[1], srcData.shape[2], srcData.shape[3]), 'data length:', srcData.dtype)

    return [srcData, tarData]




# dataset path
# file_path = 'C:/Users/msh21/PycharmProjects/GenMC-data-model/training data/labels_2604_250_500_data_training_256.mat'
# # load dataset
# [src_images, tar_images] = load_mat_data(file_path,'fcw_raw_norm_log_256_filtered','optical_mask_256_filtered')
# print('Load:', src_images.shape, tar_images.shape)
#
# # save as compressed numpy array
# filename = '/dataset_2604_500_256.npz'
# np.savez_compressed(filename, src_images, tar_images)
# print('Data saved:', filename)

def show_tensor_images(image_tensor):
    '''
    Function for visualising images in tensor format: given a tensor of images, number of images, and size
    per image, plots and prints the images in an uniform grid
    '''
    image_unflat = image_tensor.detach().cpu()
    pyplot.imshow(image_unflat[0, 0, :, :].rot90(3), cmap='viridis')
    pyplot.show()


if __name__ == '__main__':
# check if data is saved successfully
data = np.load('C:/Users/msh21/PycharmProjects/GenMC-data-model/training data/dataset_2604_500_256.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
# number of samples used for checking
n_samples = 3
n_properties = 3
for i in range(n_samples):
    ax = pyplot.subplot(n_properties+1, n_samples, 1+i)
    pyplot.axis('off')
    src_image=src_images[i]
    pyplot.imshow(np.rot90(src_image[:,:,0], k=-1))
    ax.set_title('absorption')

for i in range(n_samples):
    ax = pyplot.subplot(n_properties+1, n_samples, 1+n_samples+i)
    pyplot.axis('off')
    src_image=src_images[i]
    pyplot.imshow(np.rot90(src_image[:,:,1], k=-1))
    ax.set_title('scattering')

for i in range(n_samples):
    ax = pyplot.subplot(n_properties+1, n_samples, 1+2*n_samples+i)
    pyplot.axis('off')
    src_image=src_images[i]
    pyplot.imshow(np.rot90(src_image[:,:,2], k=-1))
    ax.set_title('Grunenisen')

for i in range(n_samples):
    ax = pyplot.subplot(n_properties+1, n_samples, 1+3*n_samples+i)
    pyplot.axis('off')
    pyplot.imshow(np.rot90(tar_images[i].squeeze(), k=-1))
    ax.set_title('optical fluence')

pyplot.show()
