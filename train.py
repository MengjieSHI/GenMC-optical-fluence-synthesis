import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from progress.bar import IncrementalBar # discard this use tqdm

from pix2pix_torch import UnetGenerator, Discriminator
from loss import GeneratorLoss, DiscriminatorLoss
from utils import show_tensor_images


# GAN (pix2pix, pix2pixHD)
def trainGAN(dataloader, models, optimizers, schedulers, device):

    generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers

    cur_step = 0
    display_step = 100

    mean_g_loss = 0.0
    mean_d_loss = 0.0

    g_loss_all = 0.0
    d_loss_all = 0.0

    epoch_g_losses = []
    epoch_d_losses = []

    for epoch in range(args.epochs):

        for index, (targets, labels) in tqdm(enumerate(dataloader), total=len(dataloader), position=0):

            targets = targets.to(device)
            labels = labels.to(device) # need to confirm whether the order is correct

            # code compressed with a collection function of loss 'loss_fn'
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device='cuda')):
                    g_loss, d_loss, x_fake = loss_fn(targets, labels, generator, discriminator)
            else:
                g_loss, d_loss, x_fake = loss_fn(targets, labels, generator, discriminator)


            # update generator
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # update discriminator
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # batch losses
            g_loss_all += g_loss.item()
            d_loss_all += d_loss.item()


        # epoch losses
        mean_g_loss = g_loss_all / (index + 1)
        #mean_g_loss = g_loss_all / len(dataloader)
        mean_d_loss = d_loss_all / (index + 1)
        #mean_d_loss = d_loss_all / len(dataloader)

        # count timeframe
        end = time.time()
        tm = (end-start)
        #logger.add_scalar('generator_loss', g_loss, epoch+1)
        #logger.add_scalar('discriminator_loss', d_loss, epoch+1)
        #logger.save_weights(generator.state_dict(), 'generator')
        print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA:%.3fs"%(epoch+1, args.epochs, mean_g_loss, mean_d_loss, tm))
        g_loss_all = 0
        d_loss_all = 0

        # this part needs optimisation
        if (epoch + 1) % 5 == 0:
            show_tensor_images(x_fake.to(targets.dtype))
            show_tensor_images(x_real)
        epoch_g_losses.append(mean_g_loss)
        epoch_g_losses.append(mean_d_loss)

        #g_scheduler.step()
        #d_scheduler.step()

    #logger.close()
    print('Training End')

# train UNET
def trainUNET(dataloader, model, optimizer, scheduler, device):

    cur_step = 0
    display_step = 100

    loss_all = 0.0
    mean_loss = 0.0
    epoch_losses = []

    for index, (targets, labels) in tqdm(enumerate(dataloader), total=len(dataloader), position=0)
        targets = targets.to(device)
        labels = labels.to(device)

        loss = 






# train diffusion models
# TBD



#
parser = argparse.ArugmentParser(prog='train', description='Train pix2pix')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
parser.add_argument("--dataset", type=str, default="256_data", help="number of datasets")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--lr", type=int, default=0.0002, help="learning rate")
parser.add_argument("--bilinear", action="store_true", default=True, help='Use bilinear upsampling')
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cup')

# loading data
dir_npz = './data128/' # the name of the folder to put all data
dataset = NPZDataset(dir_npz)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# models
print('Initise models')
generator = UnetGenerator().to(device)
discriminator = Discriminator().to(device)

# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

# loss
# loss for pix2pix
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()

# logger initialisation: defined seperately
# logger = Logger(filename=args.dataset)

unet_model = UNET(n_channels=3, n_classes=1, bilinear = args.bilinear)
unet_optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr, betas=(0.5, 0.999))
unet_criterion = torch.nn.MSELoss()
