import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, l1_loss

  
class MSE(nn.MSELoss):
    
    def __init__(self, weights=None):
        
        super(MSE, self).__init__()
        self.weights = weights
    
    def forward(self, output, target):
        
        assert not any([t.requires_grad for t in target])
        
        if isinstance(output, torch.Tensor):
            output = [output]
        
        output_len = len(output)
        weights = [1 for _ in range(output_len)] if not self.weights else self.weights
        d = torch.Tensor(weights).sum().item() if output_len > 1 else 1
        
        return torch.sum(torch.stack(
            [weights[i] / d * mse_loss(output[i], target[i]) for i in range(output_len)]
        )
                        )
    
class MAE(nn.L1Loss):
    
    def __init__(self, weights=None):
        
        super(MAE, self).__init__()
        self.weights = weights
    
    def forward(self, output, target, weights=None):
        
        assert not any([t.requires_grad for t in target])
        
        if isinstance(output, torch.Tensor):
            output = [output]
        
        output_len = len(output)
        weights = [1 for _ in range(output_len)] if not weights else weights
        d = torch.Tensor(weights).sum().item() if output_len > 1 else 1
        
        return torch.sum(torch.stack(
            [weights[i] / d * l1_loss(output[i], target[i]) for i in range(output_len)]
        )
                        )
    

class PSNRLoss(nn.MSELoss):
    
    def __init__(self, loss='mse', weights=None):
        
        super(PSNRLoss, self).__init__()
        
        if loss == 'mse':
            self.loss = mse_loss
        elif loss == 'mae':
            self.loss = l1_loss
        
        self.weights = weights
    
    
    def _psnr(self, output, target):
    
        imagewise_mse = self.loss(output, target, reduction='none').mean((1, 2, 3))
        imagewise_mse = torch.clamp(imagewise_mse, min=1e-10)
        imagewise_psnr = 10 * torch.log10(1 / imagewise_mse)

        return imagewise_psnr.mean()

    def forward(self, output, target, weights=None, validation=False):
        
        assert not any([t.requires_grad for t in target])
        
        if isinstance(output, torch.Tensor):
            output = [output]

        output_len = len(output)
        
        weights = [1 for _ in range(output_len)] if not self.weights else self.weights
        d = torch.Tensor(weights).sum().item() if output_len > 1 else 1
        
        psnr = [self._psnr(output[i], target[i]) for i in range(output_len)]
        
        if validation:
            return psnr
        else:
            return -torch.sum(torch.stack(
                [weights[i] / d * psnr[i] for i in range(output_len)]
            ))
