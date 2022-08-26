# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .bread import ConvBNReLU, norm_layer
from .ham import get_hams


class HamburgerV1(nn.Module):
    def __init__(self, in_c,n=3,D=512, args=None):
        super().__init__()

        ham_type = 'NMF'
        self.n=n

        D = getattr(args, 'MD_D', D)
    
        self.lower_bread = nn.Sequential(nn.Conv2d(in_c, D, 1),
                                         nn.ReLU(inplace=True))
    
        HAM = get_hams(ham_type)
        
        self.ham = HAM(args,D=D)
        
        
        self.upper_bread = nn.Sequential(nn.Conv2d(D, in_c, 1, bias=False),
                                         norm_layer(in_c))
        self.shortcut = nn.Sequential()
        
        self._init_weight()
        
        print('ham', HAM)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        _,c,h,w=x.size()
        shortcut = self.shortcut(x)
        x = self.lower_bread(x)
        x_c=x.size(1)
        x=x.view(-1,self.n,x_c,h,w)
        x = self.ham(x)
        x=x.contiguous().view(-1,x_c,h,w)
        x = self.upper_bread(x)
        x = F.relu(x + shortcut, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)






