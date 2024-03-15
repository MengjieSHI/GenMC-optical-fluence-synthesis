import torch
from torch import nn

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        #self.l1 = nn.L1Loss()
        self.L2 = nn.L2loss()

    def forward(self, fake, real, fake_pred):
        fake_label = torch.zeros_like(fake_pred)
        loss = self.bce(fake_pred, fake_label) + self.alpha * self.l1(fake, real)
        return loss



class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss_fn =nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred):
        fake_label = torch.zeros_like(fake_pred)
        real_label = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_label)
        real_loss = self.loss_fn(real_pred, real_label)
        loss = (fake_loss + real_loss)/2
        return loss
