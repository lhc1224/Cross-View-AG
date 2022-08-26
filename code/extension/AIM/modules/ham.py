# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args,D,R,spatial=True):
        super().__init__()

        
        self.spatial=spatial
        
        self.S = getattr(args, 'MD_S', 1)
        self.D=D
        self.R=R

       

        self.train_steps = getattr(args, 'TRAIN_STEPS', 6)
        self.eval_steps  = getattr(args, 'EVAL_STEPS', 7)

        self.inv_t = getattr(args, 'INV_T', 1)
        self.eta   = getattr(args, 'ETA', 0.9)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)

        self.bases = self._build_bases(1, self.S, self.D, self.R, cuda=True)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x,x_2, return_bases=False):
        B, Num,C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)

        
        D = C // self.S
        N = Num*H * W
        x=x.permute(0,2,1,3,4)    # [B,C,Num,H,W]

        x = x.contiguous().view(B * self.S, D, N)   #### [B,C,Num*H*W]

        
        bases=self.bases.repeat(B,1,1)  ### [B*S,D,R]
        bases, coef = self.local_inference(x, bases)
        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        x_2 = x_2.contiguous().view(B * self.S, D, W*H)   ##  [B,C,W*H]
        coef_2 = torch.bmm(x_2.transpose(1, 2), bases)
        coef_2 = F.softmax(self.inv_t * coef_2, dim=-1)
        x_2=torch.bmm(bases,coef_2.transpose(1, 2))
        # (B * S, D, N) -> (B, C, H, W)
        
        x=x.view(B,C,Num,H,W)
        x=x.permute(0,2,1,3,4)
        x_2=x_2.view(B,C,H,W)
        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)
        self.online_update(bases)
        return x,x_2

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        bases=bases
        update = bases.mean(dim=0)
        self.bases=self.bases
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)

class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args,D,R,spatial=True):
        super().__init__(args,D,R,spatial)
        self.inv_t = 1
        self.D=D
        self.R=R
        self.spatial=spatial

    def _build_bases(self, B, S,D, R, cuda=True):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()

        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef

def get_hams(key):
    hams = {'NMF':NMF2D}

    assert key in hams
    return hams[key]
