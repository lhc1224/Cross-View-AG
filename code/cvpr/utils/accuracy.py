import torch
import numpy as np 
#from utils.IoU import IoU

def compute_cls_acc(preds, label):
    pred = torch.max(preds, 1)[1]
    #label = torch.max(labels, 1)[1]
    num_correct = (pred == label).sum()
    return float(num_correct)/ float(preds.size(0))



class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0
    
    def updata(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.cnt += n
        if self.cnt == 0:
            self.avg = 1
        else:
            self.avg = self.sum / self.cnt
