<<<<<<< HEAD
import numpy
import torch
from sklearn.metrics import mean_squared_error as mse
from dataset_load import *
from numpy.random import randint
from numpy import squeeze
from statistics import mean
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

#display results on training set
#load used dataset for training

[X1, X2] = load_real_samples('test data/dataset_test_152025_256_real.npz') # wrist
print('Loaded,',X1.shape, X2.shape)

#load the saved Keras  model
model = load_model('./trained models/model_002500.h5') # current the best

psnr_val=[]
ssim_val=[]
mse_val=[]

gt=[]
gen=[]
src=[]


for i in range(0, 3):

    src_image, tar_image = X1[[i]], X2[[i]]

    gen_image = model.predict(src_image)
    # # plot the source(mask), generated image and the ground truth
    src_mua=expand_dims(src_image[:,:,:,0],axis=3)

    diff_image = gen_image-tar_image
    plot_images(src_mua, gen_image, tar_image,diff_image)

    psnr_val+=[psnr(squeeze(tar_image),squeeze(gen_image),data_range=gen_image.max()-gen_image.min())]
    ssim_val+=[ssim(squeeze(tar_image),squeeze(gen_image),data_range=gen_image.max()-gen_image.min())]
    #ssim_torch=SSIM(data_range=gen_image.max()-gen_image.min())
    #ssim_val+=[ssim_torch(torch.from_numpy(tar_image).permute(3,0,1,2),torch.from_numpy(gen_image).permute(3,0,1,2))]
    mse_val +=[mse(squeeze(tar_image),squeeze(gen_image))]


    gt.append(squeeze(tar_image))
    gen.append(squeeze(gen_image))
    src.append(squeeze(src_image))

formatted_psnr = ['%.2f'% elem for elem in psnr_val]
formatted_ssim = ['%.2f'% elem for elem in ssim_val]
formatted_mse = ['%.2f'% elem for elem in mse_val]
print('psnr:', formatted_psnr,'ssim:', formatted_ssim,'mse:', formatted_mse)
print(f'psnr_mean:{mean(psnr_val):,.2f},ssim_mean:{mean(ssim_val):,.2f},mse_mean:{mean(mse_val):,.2f}')



=======
import numpy
import torch
from sklearn.metrics import mean_squared_error as mse
from dataset_load import *
from numpy.random import randint
from numpy import squeeze
from statistics import mean
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

#display results on training set
#load used dataset for training

[X1, X2] = load_real_samples('test data/dataset_test_152025_256_real.npz') # wrist
print('Loaded,',X1.shape, X2.shape)

#load the saved Keras  model
model = load_model('./trained models/model_002500.h5') # current the best

psnr_val=[]
ssim_val=[]
mse_val=[]

gt=[]
gen=[]
src=[]


for i in range(0, 3):

    src_image, tar_image = X1[[i]], X2[[i]]

    gen_image = model.predict(src_image)
    # # plot the source(mask), generated image and the ground truth
    src_mua=expand_dims(src_image[:,:,:,0],axis=3)

    diff_image = gen_image-tar_image
    plot_images(src_mua, gen_image, tar_image,diff_image)

    psnr_val+=[psnr(squeeze(tar_image),squeeze(gen_image),data_range=gen_image.max()-gen_image.min())]
    ssim_val+=[ssim(squeeze(tar_image),squeeze(gen_image),data_range=gen_image.max()-gen_image.min())]
    #ssim_torch=SSIM(data_range=gen_image.max()-gen_image.min())
    #ssim_val+=[ssim_torch(torch.from_numpy(tar_image).permute(3,0,1,2),torch.from_numpy(gen_image).permute(3,0,1,2))]
    mse_val +=[mse(squeeze(tar_image),squeeze(gen_image))]


    gt.append(squeeze(tar_image))
    gen.append(squeeze(gen_image))
    src.append(squeeze(src_image))

formatted_psnr = ['%.2f'% elem for elem in psnr_val]
formatted_ssim = ['%.2f'% elem for elem in ssim_val]
formatted_mse = ['%.2f'% elem for elem in mse_val]
print('psnr:', formatted_psnr,'ssim:', formatted_ssim,'mse:', formatted_mse)
print(f'psnr_mean:{mean(psnr_val):,.2f},ssim_mean:{mean(ssim_val):,.2f},mse_mean:{mean(mse_val):,.2f}')



>>>>>>> bfb5f61a76607811cd56da41749da16d03d1f831
