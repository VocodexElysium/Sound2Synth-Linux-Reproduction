from utils import *

import pytorch_lightning as pl
from pyheaven.torch_utils import HeavenDataset, HeavenDataLoader
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

import numpy as np

class Sound2SyntheModel(pl.LightningModule):
    """Main model."""

    def __init__(self, net, interface, args):
        """Constructor.
        
        Args:
            net: The overall Neural Network
            interface:
            args:
        """
        super().__init__
        self.net = net
        self.interface = interface
        self.args = args
        self.criteria = interface.criteria
        self.learning_rate = args.learning_rate

    def forward(self, x):
        """Forward propagate."""
        return self.net(x)
    
    def configure_optimizers(self):
        """Set optimizers."""
        optimizer = optim.AdamW(
            self.net.parameters(),
            lr = self.args.learning_rate,
            weight_decay = self.args.weight_decay,
        )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmop_epochs = self.args.warmup_epochs,
            warmup_start_lr = self.args.warmup_start_lr_ratio * self.args.learning_ratem,
            eta_min = self.args.eta_min,
        )

        return [optimizer], [scheduler]

    def train_dataloader(self):
        """Get the dataloader of training data."""
        return HeavenDataLoader(DataLoader(
            self.args.datasets.train,
            batch_size = self.args.batch_size,
            num_workers = 8,
            shuffle = True),
            self.args.datasets.train)

    def validation_dataloader(self):
        """Get the dataloader of validation data."""
        return HeavenDataLoader(DataLoader(
            self.args.datasets.validation,
            batch_size = self.args.batch_size,
            num_workers = 8,
            shuffle = False),
            self.args.datasets.validation)

    def test_dataloader(self):
        """Get the dataloader of test data."""
        return HeavenDataLoader(DataLoader(
            self.args.datasets.test,
            batch_size = 1,
            num_workers = 8,
            shuffle = False),
            self.args.datasets.test)

    def run_batch(self, batch, split='train', batch_idx=-1):
        """"""
        