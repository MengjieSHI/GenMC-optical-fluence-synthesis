import torch
from torch import nn
import time
import argparse
from progress.bar import IncrementalBar

from pix2pix_torch import UnetGenerator, Discriminator
from loss import GeneratorLoss, DiscriminatorLoss

#
parser = argparse.ArugmentParser(prog='train', description='Train pix2pix')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
parser.add_argument("--dataset", type=str, default="256_data", help="number of datasets")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--lr", type=int, default=0.0002, help="learning rate")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cup')

# models
print('Initise models')
generator = UnetGenerator().to(device)
discriminator = Discriminator().to(device)
# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
# loss
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()
# dataset

# logger initialisation: defined seperately
# logger = Logger(filename=args.dataset)

for epoch in range(args.epochs):
    generator_loss = 0
    discriminator_loss = 0
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]',max=len(dataloader))
    for input, real_target in dataloader:
        input = input.to(device)
        real_target = real_target.to(device) # real target

        # generator loss
        fake_target = generator(input)
        fake_pred = discriminator(fake_target, input) # probability
        g_loss = g_criterion(fake_target, real_target, fake_pred)

        # discriminator loss
        #fake_target = generator(input).detach() #
        real_pred = discriminator(real_target, input)
        fake_pred = discriminator(fake_target, input)
        d_loss = d_criterion(fake_pred, real_pred)

        # update generator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # update discriminator
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # batch losses
        generator_loss += g_loss.item()
        discriminator_loss += d_loss.item()
        bar.next()
    bar.finish()
    # epoch losses
    g_loss = generator_loss/len(dataloader)
    d_loss = discriminator_loss/len(dataloader)
    # count timeframe
    end = time.time()
    tm = (end-start)
    #logger.add_scalar('generator_loss', g_loss, epoch+1)
    #logger.add_scalar('discriminator_loss', d_loss, epoch+1)
    #logger.save_weights(generator.state_dict(), 'generator')
    print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA:%.3fs"%(epoch+1, args.epochs, g_loss, d_loss, tm))
#logger.close()
print('Training End')


