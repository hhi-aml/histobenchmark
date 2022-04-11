import torch
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np

class PESG_AUC(Optimizer):
    """ optimizer from https://arxiv.org/abs/2012.03173
    params: NN params PLUS a and b as parameters of the loss
    params_alpha: alpha parameters in the loss
    lr aka eta

    """

    def __init__(self, params, lr=1e-3, ema=0.9, gamma=1e-4, weight_decay=1e-3,square_loss=False):
        defaults = dict(lr=lr, ema=ema, gamma=gamma,  weight_decay=weight_decay, is_alpha=False)
        super(PESG_AUC, self).__init__(params, defaults)
        self.square_loss = square_loss


    def __setstate__(self, state):
        super(PESG_AUC, self).__setstate__(state)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lr = None
        for group in self.param_groups:

            weight_decay = group['weight_decay']
            gamma = group['gamma']
            ema = group['ema']
            lr = group['lr']
            is_alpha = group['is_alpha']

            for p in group['params']:
                if p.grad is not None:
                
                    if(not(is_alpha)):#all other params update    
                        if(gamma !=0):
                            param_state = self.state[p]
                            if 'avg_param_buffer' not in param_state:
                                ref = param_state['avg_param_buffer'] = torch.clone(p).detach()
                            else:
                                ref = param_state['avg_param_buffer']
                                ref.mul_(1-ema).add_(p, alpha=ema)
                        d_p = p.grad
                        if weight_decay != 0:
                            d_p = d_p.add(p, alpha=weight_decay)
                        if gamma != 0:
                            d_p = d_p.add(gamma* (p-ref))
                
                        p.add_(d_p, alpha=-lr)
                    else:#alpha update
                        p.add_(p.grad*lr)
                        if(not(self.square_loss)):
                            p = p.clip(min=0)
        return loss

        
class auc_loss(nn.Module):
    '''loss from https://arxiv.org/abs/2012.03173 c.f. also https://github.com/yzhuoning/Deep_AUC/blob/master/optimizer.py
    note: different paramgroup for alpha with param is_alpha=True
    can either pass an int as ps (number of classes) or a list of the prior class probabilities'''
    def __init__(self,ps,m=0.5, sigmoid=False, reduction='mean', square_loss=False):
        super(auc_loss, self).__init__()
        self.use_counts = isinstance(ps,int)
        num_classes = ps if self.use_counts else len(ps)
        
        self.a = nn.Parameter(0.01*torch.ones(num_classes))# if len(ps)>2 else 1))
        self.b = nn.Parameter(0.01*torch.ones(num_classes))# if len(ps)>2 else 1))
        self.alpha = nn.Parameter(0.01*torch.ones(num_classes))# if len(ps)>2 else 1)) #ideal 1+b-a

        if(self.use_counts):
            self.register_buffer("counts",torch.zeros(num_classes+1))
        else:
            self.register_buffer("ps", torch.from_numpy(np.array(ps)))
        
        self.m = 1 if square_loss else m
        self.sigmoid = sigmoid
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if(self.use_counts):
            self.counts[0] += target.size(0)
            self.counts[1:] += target.sum(dim=0)
            ps = self.counts[1:]/self.counts[0]
        else:
            ps = self.ps
        
        if(self.sigmoid):
            input = torch.sigmoid(input)
        else:
            input = torch.softmax(input, dim=1)
        if len(target.shape) == 1:
            C = input.shape[1]
            device = target.device
            encodings = []
            for i in target.detach().cpu().numpy():
                enc = np.arange(start=0, stop=C, step=1) == i
                encodings.append(enc)
            target = torch.from_numpy(np.array(encodings)).to(device)

        A1 = (1-ps)*(input-self.a).pow(2)
        A1 = A1*(target==1)
        A2 = ps*(input-self.b).pow(2)*(target==0)
        A3 = -(1-ps)*ps*self.alpha*self.alpha
        A4 = 2*self.alpha*(ps*(1-ps)*self.m +ps*input*(target==0) - (1-ps)*input*(target==1))
        loss = torch.mean((A1 + A2 + A3 + A4),dim=1)#mean over classes ## (target.sum(dim=0)>0).float()*
        #reduction over samples
        if(self.reduction=="mean"):
            return torch.mean(loss)
        elif(self.reduction=="sum"):
            return torch.sum(loss)
        elif(self.reduction=="none"):
            return loss
        assert(True)#invalid reduction
