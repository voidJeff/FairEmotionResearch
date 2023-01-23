"""
Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class baseline_pretrain(nn.Module):
    """
    Baseline model for facial expression prediction

    Finetunes a layer on top of resnet pretrain

    Args:
        hidden_size,
        drop_prob
    """

    def __init__(self, in_features,  num_classes = 8):
        super(baseline_pretrain, self).__init__()
        self.in_features = in_features

        self.model_ft= resnet50(pretrained = ResNet50_Weights)
        self.fc1 = nn.Linear(in_features, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, num_classes)



    def forward(self, x):

        # forward through linear layers
        out = self.fc1(x)
        out = F.relu(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out
