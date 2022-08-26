import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def default_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std

def load_img(fl):
    return Image.open(fl).convert('RGB')

def batch_cuda(batch):
    return {k:v.cuda() if type(v)==torch.Tensor else v for k,v in batch.items()}

def unnormalize(tensor):
    mean, std = default_mean_std()
    u_tensor = tensor.clone()

    def _unnorm(t):
        for c in range(3):
            t[c].mul_(std[c]).add_(mean[c])

    if u_tensor.dim()==4:
        [_unnorm(t) for t in u_tensor]
    else:
        _unnorm(u_tensor)
    
    return u_tensor

def default_transform(split):

    mean, std = default_mean_std()

    if split=='train':
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
    else:
        transform = transforms.Compose([
                        #transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
            ])

    return transform


import utils.gtransforms as gtransforms
def clip_transform(split, max_len):

    mean, std = default_mean_std()

    if split=='train':
        transform = transforms.Compose([
                        gtransforms.GroupResize(256),
                        gtransforms.GroupRandomCrop(224),
                        gtransforms.GroupRandomHorizontalFlip(),
                        gtransforms.ToTensor(),
                        gtransforms.GroupNormalize(mean, std),
                        gtransforms.ZeroPad(max_len),
                    ])

    elif split=='val':
        transform = transforms.Compose([
                        gtransforms.GroupResize(256),
                        gtransforms.GroupCenterCrop(224),
                        gtransforms.ToTensor(),
                        gtransforms.GroupNormalize(mean, std),
                        gtransforms.ZeroPad(max_len),
            ])

    return transform


import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numbers
class PairedTransform:

    def __init__(self, split, std_norm=True):
        self.split = split
        self.mean, self.std = default_mean_std()
        self.std_norm = std_norm

    def train_transform(self, image, heatmap):

        heatmap = torch.from_numpy(heatmap)

        # PIL and Tensor have inverted shapes
        # image.size == (606, 479) while heatmap.shape == (479, 606)
        assert (heatmap.shape[1], heatmap.shape[0]) == image.size, 'image and heatmap sizes mismatch (%s vs. %s)'%(image.size, heatmap.shape)

        # resize
        image = TF.resize(image, size=(256, 256))
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=image.size, mode='bilinear', align_corners=False)[0][0]

        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        heatmap = heatmap[i:i+w, j:j+h]

        # horizontal flip
        if np.random.rand()<0.5:
            image = TF.hflip(image)
            heatmap = heatmap.flip(1)

        # to tensor + normalize
        image = TF.to_tensor(image)

        if self.std_norm:
            image = TF.normalize(image, self.mean, self.std)

        if heatmap.sum().item()!=0:
            heatmap = heatmap/heatmap.sum()

        return image, heatmap

    def val_transform(self, image, heatmap):

        # PIL and Tensor have inverted shapes
        # image.size == (606, 479) while heatmap.shape == (479, 606)
        assert (heatmap.shape[1], heatmap.shape[0]) == image.size, 'image and heatmap sizes mismatch (%s vs. %s)'%(image.size, heatmap.shape)

        heatmap = torch.from_numpy(heatmap)

        # resize
        image = TF.resize(image, size=(224, 224))
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=image.size, mode='bilinear', align_corners=False)[0][0]

        # to tensor + normalize
        image = TF.to_tensor(image)

        if self.std_norm:
            image = TF.normalize(image, self.mean, self.std)

        if heatmap.sum().item()!=0:
            heatmap = heatmap/heatmap.sum()

        return image, heatmap

    def __call__(self, image, heatmap):

        if self.split == 'train':
            image, heatmap = self.train_transform(image, heatmap)
        elif self.split =='val':
            image, heatmap = self.val_transform(image, heatmap)

        return image, heatmap