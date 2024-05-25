import os
import numpy as np
import cv2
import torch
gt_root="dataset/testsetv2/Seen/testset/GT"
files=os.listdir(gt_root)
dict_1={}
for file in files:
    file_path=os.path.join(gt_root,file)
    objs=os.listdir(file_path)
    for obj in objs:
        obj_path=os.path.join(file_path,obj)
        images=os.listdir(obj_path)
        for img in images:
            img_path=os.path.join(obj_path,img)
            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #mask=mask/255.0
            key=file+"_"+obj+"_"+img
            dict_1[key]=mask
torch.save(dict_1,"Seen_gt.t7")
