import os
import argparse
from re import T
import torch
import torch.nn as nn
from models import *
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from utils.accuracy import *
from utils.lr import *
import random
import cv2
import torch.nn.functional as F
from utils.evaluation import cal_kl,cal_sim,cal_nss
import time
##  set seed
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='dataset/AGD20K')
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument("--divide",type=str,default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
##  dataloader
parser.add_argument('--num_workers', type=int, default=8)
##  train
parser.add_argument('--batch_size', type=int, default=14)
parser.add_argument('--T', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=38)
parser.add_argument('--pretrain', type=str, default='True')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--power', type=float, default=0.9)
parser.add_argument('--init_weights', type=bool, default=True)
parser.add_argument('--nest', action='store_true')
parser.add_argument("--backbone",type=str,default="resnet50")  ### resnet50
##  model
parser.add_argument('--mode', type=str, default='base')
parser.add_argument("--arch",type=str,default="resnet_AIM")
parser.add_argument("--distil_w",type=float,default=0.5)
parser.add_argument("--align_w",type=float,default=0.5)
parser.add_argument("--cls_w",type=float,default=1.0)
parser.add_argument("--pretrained",type=str,default=True)
parser.add_argument("--n",type=int,default=3)
parser.add_argument("--D",type=int,default=256)
parser.add_argument("--R",type=int,default=64)
##  show
parser.add_argument('--show_step', type=int, default=50)
##  GPU'
parser.add_argument('--gpu', type=str, default='0')

#### test
parser.add_argument("--test_batch_size",type=int,default=1)
parser.add_argument('--test_num_workers', type=int, default=1)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
lr = args.lr
if args.divide=="Seen":   #### Seen
    args.num_classes=36
    
else:
    args.num_classes=25   #### Unseen

args.test_root=os.path.join(args.data_root,args.divide,"testsetv2","egocentric")
args.mask_root=os.path.join(args.data_root,args.divide,"testsetv2","GT")

def img_processing(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    img = Variable(img).cuda()
    return img


def normalize_map(atten_map):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)
    return atten_norm


if args.divide == "Seen":
    args.num_classes = 36
    aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                "talk_on", "text_on", "throw", "type_on", "wash", "write"]
else:
    args.num_classes = 25
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                "swing", "take_photo", "throw", "type_on", "wash"]
args.test_root = os.path.join(args.data_root, args.divide,"testsetv2", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide,"testsetv2", "GT")

model_list=os.path.join(args.model_file)
for model_path in model_list:
    if model_path[-7:]!="pth.tar":
        continue
    pred_path=os.path.join(args.model_file,args.divide+"_"+model_path[-8:]+"_preds.t7")
    if not os.path.exists(pred_path):
        model=eval(args.arch).model(args)
        model.cuda()
        model.load_state_dict(torch.load(os.path.join(args.model_file,model_path)))
        model.eval()
        model.cuda()
        from data_load.datatest import TestData
        testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)

        MyDataLoader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=args.test_batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=False)

        dict_1 = {}
    
        for step, (image, label, mask_path) in enumerate(MyDataLoader):
            output = model.test_forward(image.cuda())
            cam = model.get_fused_cam(target=label.long().cuda())[0].data.cpu()
            cam = np.array(cam)
            names = mask_path[0].split("/")
            key = names[-3] + "_" + names[-2] + "_" + names[-1]
            dict_1[key] = cam
        torch.save(dict_1,pred_path)
    masks = torch.load(args.divide+"_gt.t7")
    preds=torch.load(pred_path)

    KLs = [] 
    SIMs = []
    NSSs = []

    for key in masks.keys():
        print(key)
        gt=masks[key]
        mask=gt/255.0
        pred=preds[key]
        if pred.shape!=mask.shape:
            pred=cv2.resize(pred,(mask.shape[1],mask.shape[0]))
        pred=(pred-pred.min())/(pred.max()-pred.min()+1e-12)
        kld = cal_kl(pred, mask)
        sim=cal_sim(pred,mask)
        nss=cal_nss(pred,mask)

        KLs.append(kld)
        SIMs.append(sim)
    
        NSSs.append(nss)
    mKLD=sum(KLs)/len(KLs)
    mSIM=sum(SIMs)/len(SIMs)
    mNSS=sum(NSSs)/len(NSSs)
    print("kld = "+str(sum(KLs)/len(KLs)))
    print("sim = "+str(sum(SIMs)/len(SIMs)))
    print("nss = "+str(sum(NSSs)/len(NSSs)))
    str_2=model_path+":\n"+"mKLD = "+str(mKLD)+"\n"+ \
        "mSIM = "+str(mSIM)+"\n"+\
        "mNSS = "+str(mNSS)+"\n"

    with open(os.path.join(args.model_file,"model_result.txt"),"a") as f:
        f.write(str_2)
    f.close()



