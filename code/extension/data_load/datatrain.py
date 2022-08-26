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
from torchvision import transforms


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
class TrainData(data.Dataset):
    def __init__(self, exocentric_root, egocentric_root,resize_size=256,crop_size=224,divide="Seen",n=3):
        
        self.exocentric_root = exocentric_root
        self.egocentric_root = egocentric_root

        self.image_list = []
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.n=n
        if divide=="Seen":
            self.aff_list = ['beat',"boxing","brush_with","carry","catch","cut","cut_with","drag",'drink_with',
                       "eat","hit","hold","jump","kick","lie_on", "lift","look_out","open","pack","peel",
                       "pick_up","pour","push","ride","sip","sit_on","stick","stir","swing","take_photo",
                       "talk_on","text_on","throw","type_on","wash", "write"]
        else:
            self.aff_list=["carry","catch","cut","cut_with",'drink_with',
                       "eat","hit","hold","jump","kick","lie_on","open","peel",
                       "pick_up","pour","push","ride","sip","sit_on","stick",
                       "swing","take_photo", "throw","type_on","wash"]

        self.transform = transforms.Compose([
                         transforms.Resize(resize_size),
                         transforms.RandomCrop(crop_size),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std)
                     ])

        files = os.listdir(self.exocentric_root)
        for file in files:
            file_path = os.path.join(self.exocentric_root, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path=os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    img_path = os.path.join(obj_file_path, img)
                    self.image_list.append(img_path)
        

    def __getitem__(self, item):
        exocentric_image_path=self.image_list[item]
        names = exocentric_image_path.split("/")
        aff_name,object = names[-3], names[-2]
        object_file = os.path.join(self.egocentric_root, aff_name, object)
        label = self.aff_list.index(aff_name)
        obj_images = os.listdir(object_file)
        idx = random.randint(0, len(obj_images)-1)
        egocentric_image_path=os.path.join(object_file,obj_images[idx])
        egocentric_image = self.load_static_image(egocentric_image_path)


        exocentric_file=os.path.join(self.exocentric_root,aff_name,object)
        exocentrics = os.listdir(exocentric_file)

        exocentric_images=[]
        if len(exocentrics)<self.n:
            start=0
            for i in range(start,len(exocentrics)):
                tmp_exo=self.load_static_image(os.path.join(self.exocentric_root,aff_name,object,exocentrics[i]))
                exocentric_images.append(tmp_exo)
            for i in range(len(exocentrics),self.n):
                exocentric_images.append(tmp_exo)
        else:
            
            start=random.randint(0,len(exocentrics)-self.n)
            for idx in range(start,start+self.n):
                tmp_exo=self.load_static_image(os.path.join(self.exocentric_root,aff_name,object,exocentrics[idx]))
                exocentric_images.append(tmp_exo)

        return exocentric_images,egocentric_image,label

    def load_static_image(self, path):

        img = util.load_img(path)
        img = self.transform(img)
        return img


    def __len__(self):

        return len(self.image_list)


