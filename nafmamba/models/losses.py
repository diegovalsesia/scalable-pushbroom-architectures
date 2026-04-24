import torch
import torch.nn as nn
import numpy as np
from nafmamba.models.metrics import mse
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class STDLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(STDLoss, self).__init__()
        self.weight = loss_weight
        self.scale = 10 / np.log(10)
    def forward(self,outputs):
        output_tensor = torch.stack([t for t in outputs])
        std = torch.std(output_tensor)
        return torch.log(std+ 1e-8)* self.weight *self.scale
    
class  ScalableLoss(nn.Module):
    def __init__(self, base_loss_weight=1.0, margin=0.1, temp=0.5):
        super().__init__()
        self.base_loss_weight = base_loss_weight
        self.margin = margin  
        self.temp = temp   

    def forward(self, outputs, target):
        loss = 0
        for i in range(1, len(outputs)):
            mse_decrease =  mse(outputs[i - 1], target) - mse(outputs[i], target) 
            loss += mse_decrease
        return loss * self.base_loss_weight
    
class UncertaintyLoss(nn.Module):
    def __init__(self, base_loss_weight=1.0):
        super().__init__()
        self.base_loss_weight = base_loss_weight


    def forward(self, output, output_features,target):
        variance = torch.var(torch.stack(output_features,dim=0))
        loss = variance + torch.exp(-variance) * mse(output, target)
        return loss * self.base_loss_weight