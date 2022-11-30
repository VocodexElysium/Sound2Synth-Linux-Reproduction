import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DistributionalCrossEntropyDistance(nn.Module):
    """Calculate the distributional cross entropy distance."""
    
    def forward(self, pred, true):
        return -torch.sum(true * torch.log(torch.clip(F.softmax(pred, dim=-1), 1e-9, 1)), dim=-1) / (np.log(true.shape[-1]))

class SmoothedCrossEntropyDistance(nn.Module):
    """The 'smoothed cross entropy' mentioned in the paper."""
     
    def __init__(self, k=5, sigma=0.0):
        """Initializer.
        
           Args:
               k: The number of segments.
               sigma: The standarad deviation used in the gaussian kernel.
        """
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.cache = {}

    def kernel(self, n):
        """Calculate the Gaussian kernel of σ = σ0/K."""
        if n not in self.cache:
            tmp_kernel = torch.exp(-(torch.arange(-self.k, self.k + 1) / n / self.sigma) ** 2 / 2)
            self.cache[n] = tmp_kernel
            self.cache[n] /= self.cache[n].sum()
        return self.cahce[n]

    def weight(self, true):
        """Calculate the target for cross entropy loss computation."""
        #Create a 1x1 kernel.
        tmp_kernel = self.kernel(true.shape[-1]).unsqueeze(0).unsqueeze(1).type_as(true)
        #1d-convoluted with a Gaussian kernel of σ = σ0/K.
        weights = F.conv1d(true.unsqueeze(1), tmp_kernel, padding='same')
        #The result is normalized to have a total probability mass sum to 1.
        result = F.normalize(weights.squeeze(1), p=1, dim=-1)
        return result

    def forward(self, pred, true):
        """Calculate the cross entropy loss."""
        tmp_true = self.weight(true).detach()
        return -torch.sum(tmp_true * torch.log(torch.clip(F.softmax(pred, dim=-1), 1e-9, 1)), dim=-1) / np.log(true.shape[-1])

class ClassificationLpDistance(nn.Module):
    """Calculate the classification Lp distance."""

    def __init__(self, p=2.0):
        super().__init__()
        self.p = p

    def forward(self, pred, true):
        n = true.shape[-1]
        Z = (n - 1) * (1 / n) ** self.p + (1 - 1 / n) ** self.p
        return torch.sum((F.softmax(pred, dim=-1) - true).abs() ** self.p, dim=-1) / Z

class ClassificationAccuracy(nn.Module):
    """Calculate the claasification accuracy."""
    
    def forward(self, pred, true):
        return torch.eq(pred.argmax(dim=-1), true.argmax(dim=-1)).float()

class RegressionLpDistance(nn.Module):
    """Calculate the regression Lp distance."""

    def __init__(self, p=2.0):
        super().__init__()
        self.p = p

    def forward(self, pred, true):
        return (pred.argmax(dim=-1) - true.argmax(dim=-1)).abs() ** self.p / (true.shape[-1] - 1)

LOSSES_MAPPING = {
    'maeloss': {
        'regl': ClassificationLpDistance(p=1.0),
        'clsl': ClassificationLpDistance(p=1.0),
    },
    'mseloss': {
        'regl': ClassificationLpDistance(p=2.0),
        'clsl': ClassificationLpDistance(p=2.0),
    },
    'celoss': {
        'regl': SmoothedCrossEntropyDistance(sigma=0.02),
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'regceloss': {
        'regl': SmoothedCrossEntropyDistance(sigma=0.02),
        'clsl': None,
    },
    'clsceloss': {
        'regl': None,
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'mixloss': {
        'regl': RegressionLpDistance(p=1.0),
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'clsacc': {
        'regl': None,
        'clsl': ClassificationAccuracy(),
    },
    'regacc': {
        'regl': ClassificationAccuracy(),
        'clsl': None,
    },
    'regmae': {
        'regl': RegressionLpDistance(p=1.0),
        'clsl': None,
    }
}