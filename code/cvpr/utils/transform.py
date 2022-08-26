import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random
import json

def load_image(pah,image_size=320,if_resize=False):

    if not os.path.exists(pah):
        print(pah)
        print('File Not Exists')

    im = cv2.imread(pah)[:,:,::-1]
    if if_resize:
        im = cv2.resize(im, (image_size,image_size))
    #im = randomGaussianBlur(im)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_.transpose((2, 0, 1))
    return in_

def Normalize(image,mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]):
    image /= 255.0
    image=image.transpose((1,2,0))
    image-=np.array((mean[0],mean[1],mean[2]))
    image/=np.array((std[0],std[1],std[2]))

    image=image.transpose((2,0,1))

    return image

def load_image_test(pah,if_resize=False,image_size=320):
    if not os.path.exists(pah):
        print(pah)
        print('File Not Exists')

    im = cv2.imread(pah)
    if if_resize:
        im = cv2.resize(im, (image_size,image_size))
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])

    in_ = in_.transpose((2, 0, 1))
    return in_, im_size

def load_label(pah,image_size=320):
    """
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    #im = im.resize((image_size,image_size))
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    # label = cv2.resize(label, im_sz, interpolation=cv2.INTER_NEAREST)
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
        edge = edge[:, :, ::-1].copy()
    return img, label, edge

def cv_random_crop_flip(img, depth,object_image,resize_size, crop_size, random_flip=True):
    def get_params(img_size, output_size):
        h, w = img_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    if random_flip:
        flip_flag = random.randint(0, 1)
    img = img.transpose((1, 2, 0))  # H, W, C
    depth = depth.transpose((1, 2, 0))
    object_image=object_image.transpose((1,2,0))

    img = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
    object_image = cv2.resize(object_image, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)


    i, j, h, w = get_params(resize_size, crop_size)
    img = img[i:i + h, j:j + w, :].transpose((2, 0, 1))  # C, H, W
    depth = depth[i:i + h, j:j + w, :].transpose((2, 0, 1))  # C, H, W
    object_image = object_image[i:i + h, j:j + w, :].transpose((2, 0, 1))  # C, H, W

    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        depth = depth[:, :, ::-1].copy()
        object_image = object_image[:, :, ::-1].copy()

    return img, depth,object_image

def cv_random_crop_flip_ref(img, obj_mask,per_mask,resize_size, crop_size, random_flip=True):
    def get_params(img_size, output_size):
        h, w = img_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    if random_flip:
        flip_flag = random.randint(0, 1)
    img = img.transpose((1, 2, 0))  # H, W, C

    img = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
    obj_mask = cv2.resize(obj_mask, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
    per_mask = cv2.resize(per_mask, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)

    i, j, h, w = get_params(resize_size, crop_size)
    img = img[i:i + h, j:j + w, :].transpose((2, 0, 1))  # C, H, W
    obj_mask = obj_mask[i:i + h, j:j + w][np.newaxis, ...]  # 1, H, W
    per_mask = per_mask[i:i + h, j:j + w][np.newaxis, ...]

    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        obj_mask = obj_mask[:, :, ::-1].copy()
        per_mask=per_mask[:, :, ::-1].copy()
    obj_mask = obj_mask[0, :, :]  # H, W
    per_mask=per_mask[0,:,:]
    return img, obj_mask,per_mask

def cv_center_crop(img, label, edge_label,resize_size, crop_size, random_flip=True):
    def get_params(img_size, output_size):
        h, w = img_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = (h - th) / 2
        j = (w - tw) / 2
        return i, j, th, tw

    img = img.transpose((1, 2, 0))  # H, W, C
    label = label[0, :, :]  # H, W
    img = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
    edge_label=cv2.resize(edge_label,(resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
    i, j, h, w = get_params(resize_size, crop_size)
    img = img[i:i + h, j:j + w, :].transpose((2, 0, 1))  # C, H, W
    label = label[i:i + h, j:j + w][np.newaxis, ...]  # 1, H, W
    edge_label=edge_label[i:i + h, j:j + w][np.newaxis, ...]

    return img, label,edge_label

def random_crop(img, label, edge_label, size, padding=None, pad_if_needed=True, fill_img=(123, 116, 103), fill_label=0,
                padding_mode='constant'):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    if padding is not None:
        img = F.pad(img, padding, fill_img, padding_mode)
        label = F.pad(label, padding, fill_label, padding_mode)
        edge_label=F.pad(edge_label,padding,fill_label, padding_mode)

    # pad the width if needed
    if pad_if_needed and img.size[0] < size[1]:
        img = F.pad(img, (int((1 + size[1] - img.size[0]) / 2), 0), fill_img, padding_mode)
        label = F.pad(label, (int((1 + size[1] - label.size[0]) / 2), 0), fill_label, padding_mode)
        edge_label=F.pad(edge_label,(int((1 + size[1] - label.size[0]) / 2), 0), fill_label, padding_mode)

    # pad the height if needed
    if pad_if_needed and img.size[1] < size[0]:
        img = F.pad(img, (0, int((1 + size[0] - img.size[1]) / 2)), fill_img, padding_mode)
        label = F.pad(label, (0, int((1 + size[0] - label.size[1]) / 2)), fill_label, padding_mode)
        edge_label=F.pad(edge_label,(0, int((1 + size[0] - label.size[1]) / 2)), fill_label, padding_mode)

    i, j, h, w = get_params(img, size)
    return [F.crop(img, i, j, h, w), F.crop(label, i, j, h, w)],[F.crop(edge_label,i,j,h,w)]

def random_crop_ref(img, obj_mask, per_mask, size, padding=None, pad_if_needed=True, fill_img=(123, 116, 103), fill_label=0,
                padding_mode='constant'):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    if padding is not None:
        img = F.pad(img, padding, fill_img, padding_mode)
        obj_mask = F.pad(obj_mask, padding, fill_label, padding_mode)
        per_mask=F.pad(per_mask,padding,fill_label, padding_mode)

    # pad the width if needed
    if pad_if_needed and img.size[0] < size[1]:
        img = F.pad(img, (int((1 + size[1] - img.size[0]) / 2), 0), fill_img, padding_mode)
        obj_mask = F.pad(obj_mask, (int((1 + size[1] - obj_mask.size[0]) / 2), 0), fill_label, padding_mode)
        per_mask = F.pad(per_mask,(int((1 + size[1] - per_mask.size[0]) / 2), 0), fill_label, padding_mode)

    # pad the height if needed
    if pad_if_needed and img.size[1] < size[0]:
        img = F.pad(img, (0, int((1 + size[0] - img.size[1]) / 2)), fill_img, padding_mode)
        obj_mask = F.pad(obj_mask, (0, int((1 + size[0] - obj_mask.size[1]) / 2)), fill_label, padding_mode)
        per_mask = F.pad(per_mask, (0, int((1 + size[0] - per_mask.size[1]) / 2)), fill_label, padding_mode)

    i, j, h, w = get_params(img, size)
    return [F.crop(img, i, j, h, w), F.crop(obj_mask, i, j, h, w)],[F.crop(per_mask,i,j,h,w)]


def center_crop(img, label, edge_label, size, padding=None, pad_if_needed=True, fill_img=(123, 116, 103), fill_label=0,
                padding_mode='constant'):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h-th)/2
        j = (w-tw)/2
        return i, j, th, tw

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    if padding is not None:
        img = F.pad(img, padding, fill_img, padding_mode)
        label = F.pad(label, padding, fill_label, padding_mode)
        edge_label=F.pad(edge_label,padding,fill_label, padding_mode)

    # pad the width if needed
    if pad_if_needed and img.size[0] < size[1]:
        img = F.pad(img, (int((1 + size[1] - img.size[0]) / 2), 0), fill_img, padding_mode)
        label = F.pad(label, (int((1 + size[1] - label.size[0]) / 2), 0), fill_label, padding_mode)
        edge_label=F.pad(edge_label,(int((1 + size[1] - label.size[0]) / 2), 0), fill_label, padding_mode)

    # pad the height if needed
    if pad_if_needed and img.size[1] < size[0]:
        img = F.pad(img, (0, int((1 + size[0] - img.size[1]) / 2)), fill_img, padding_mode)
        label = F.pad(label, (0, int((1 + size[0] - label.size[1]) / 2)), fill_label, padding_mode)
        edge_label=F.pad(edge_label,(0, int((1 + size[0] - label.size[1]) / 2)), fill_label, padding_mode)

    i, j, h, w = get_params(img, size)
    return [F.crop(img, i, j, h, w), F.crop(label, i, j, h, w)],[F.crop(edge_label,i,j,h,w)]

