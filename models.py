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
    
    def __init__(self, num_classes):
        super(baseline_pretrain, self).__init__()

        self.model_ft= resnet50(pretrained = ResNet50_Weights)
        self.model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
        
    def forward(self, x):

        # forward through linear layers
        out = self.fc1(x)
        out = self.relu1(out)      
        out = self.relu2(self.fc2(out))  
        out = self.dropout(out)
        
        out = self.fc3(out)

        return out
