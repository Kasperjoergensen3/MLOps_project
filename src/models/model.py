# make template for pytorch lightning model

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams

        # define model layers
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        # forward pass
        batch_size, channels, width, height = x.size()

        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        # training step
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation step
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        return loss
