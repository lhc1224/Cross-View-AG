import os
from torch.utils import data
import collections
import torch
import numpy as np
import h5py
from joblib import Parallel, delayed
import random
from utils import util
import cv2
from utils.transform import Normalize,cv_random_crop_flip,load_image
from torchvision import transforms
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
class TestData(data.Dataset):
    def __init__(self, image_root,crop_size=224,divide="Seen",mask_root=None):
        self.image_root=image_root
        self.image_list = []
        self.crop_size = crop_size
        self.mask_root=mask_root
        if divide=="Seen":
            self.aff_list = ['beat',"boxing","brush_with","carry","catch",
                             "cut","cut_with","drag",'drink_with',"eat",
                             "hit","hold","jump","kick","lie_on", "lift",
                             "look_out","open","pack","peel","pick_up",
                             "pour","push","ride","sip","sit_on","stick",
                             "stir","swing","take_photo","talk_on","text_on",
                             "throw","type_on","wash", "write"]
        else:
            self.aff_list=["carry","catch","cut","cut_with",'drink_with',
                       "eat","hit","hold","jump","kick","lie_on","open","peel",
                       "pick_up","pour","push","ride","sip","sit_on","stick",
                       "swing","take_photo", "throw","type_on","wash"]

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])


        files = os.listdir(self.image_root)
        for file in files:
            file_path = os.path.join(self.image_root, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path=os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    img_path = os.path.join(obj_file_path, img)
                    mask_path=os.path.join(self.mask_root,file,obj_file,img[:-3]+"png")

                    if os.path.exists(mask_path):
                        self.image_list.append(img_path)
        #print(self.image_list)

    def __getitem__(self, item):

        image_path=self.image_list[item]
        img = Image.open(image_path).convert("RGB")
        image = self.transform(img)

        label= self.aff_list.index(image_path.split("/")[-3])
        names=image_path.split("/")


        mask_path=os.path.join(self.mask_root,names[-3],names[-2],names[-1][:-3]+"png")
        #print(os.path.exists(mask_path))

        return image,label,mask_path



    def __len__(self):

        return len(self.image_list)
