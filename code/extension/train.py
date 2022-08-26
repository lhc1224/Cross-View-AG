import os
import argparse
from re import T
import torch
import torch.nn as nn
from models import *

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
seed =0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/home2/lhc/Cross-View-AG/dataset/AGD20K')
parser.add_argument('--save_root', type=str, default='save_models/')
parser.add_argument("--divide",type=str,default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
##  dataloader
parser.add_argument('--num_workers', type=int, default=8)
##  train
parser.add_argument('--batch_size', type=int, default=14)
parser.add_argument('--T', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=35)
parser.add_argument('--pretrain', type=str, default='True')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--power', type=float, default=0.9)
parser.add_argument('--init_weights', type=bool, default=True)
parser.add_argument('--nest', action='store_true')
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
parser.add_argument('--test_num_workers', type=int, default=8)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
lr = args.lr
if args.divide=="Seen":   #### Seen
    args.num_classes=36
    
else:
    args.num_classes=25   #### Unseen

args.exocentric_root=os.path.join(args.data_root,args.divide,"trainset","exocentric")
args.egocentric_root=os.path.join(args.data_root,args.divide,"trainset","egocentric")
args.test_root=os.path.join(args.data_root,args.divide,"testset","egocentric")
args.mask_root=os.path.join(args.data_root,args.divide,"testset","GT")
time_str=time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
args.save_path=os.path.join(args.save_root,time_str)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
dict_args=vars(args)

str_1=""
for key,value in dict_args.items():
    str_1+=key+"="+str(value)+"\n"
with open(os.path.join(args.save_path,"write.txt"),"a") as f:
    f.write(str_1)

f.close()
if not os.path.exists(os.path.join(args.save_path,"logs")):
    os.makedirs(os.path.join(args.save_path,"logs"),exist_ok=True)
logger = SummaryWriter(log_dir=os.path.join(args.save_path,"log"))
def normalize_map(atten_map):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)
    atten_norm = cv2.resize(atten_norm, dsize=(args.crop_size,args.crop_size))
    return atten_norm


def get_optimizer(model, args):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list = []
    for name, value in model.named_parameters():
        if 'fc' in name:
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)

        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)
    optmizer = torch.optim.SGD([{'params': weight_list,
                                 'lr': lr},
                                {'params': bias_list,
                                 'lr': lr * 2},
                                {'params': last_weight_list,
                                 'lr': lr * 10},
                                {'params': last_bias_list,
                                 'lr': lr * 20}],
                               momentum=args.momentum,
                               weight_decay=args.weight_decay,
                               nesterov=True)
    return optmizer

if __name__ == '__main__':
    if args.phase == 'train':

        from data_load.datatrain import TrainData

        trainset = TrainData(exocentric_root=args.exocentric_root, 
        egocentric_root=args.egocentric_root,
        resize_size=args.resize_size,crop_size=args.crop_size,divide=args.divide)

        MyDataLoader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
        
        from data_load.datatest import TestData
        testset=TestData(image_root=args.test_root,
                         crop_size=args.crop_size,
                        divide=args.divide,mask_root=args.mask_root)
        MytestDataLoader = torch.utils.data.DataLoader(dataset=testset,
                                                   batch_size=args.test_batch_size,
                                                   shuffle=False,
                                                   num_workers=args.test_num_workers,
                                                   pin_memory=True)
        ##  model
        model=eval(args.arch).model(args,pretrained=args.pretrained)
        model.cuda()
        model.train()
        ##  optimizer
        optimizer = get_optimizer(model, args)

        ##  Loss
        loss_func = nn.CrossEntropyLoss().cuda()
        epoch_loss = 0
        correct = 0
        total = 0
        acc_sum = 0
        a = 0
        best_kld=1000
        log_step_interval = 50
        print('Train begining!')
        for epoch in range(0, args.epochs):
            model.train()
            poly_lr_scheduler(optimizer, epoch,decay_epoch=33)
            ##  accuracy
            exo_acc = AverageMeter()
            ego_acc = AverageMeter()
            epoch_loss = 0

            aveGrad = 0
            for step, (exocentric_image, egocentric_image, label) in enumerate(MyDataLoader):
                global_iter_num = epoch * len(MyDataLoader) + step + 1
                label = label.cuda(non_blocking=True)
                egocentric_image =  egocentric_image.cuda(
                    non_blocking=True)
                ##  backward
                optimizer.zero_grad()
                exocentric_images = Variable(torch.stack(exocentric_image, dim=1)).cuda(non_blocking=True)

                exo_out, ego_out = model(exocentric_images, egocentric_image)
                label = label.long()
                loss, cls_loss, dist_loss,exo_cls_loss, ego_cls_loss = model.get_loss(label,T=args.T)
                exo_cls_acc = 100. * compute_cls_acc(exo_out, label)
                ego_cls_acc = 100. * compute_cls_acc(ego_out, label)
                cur_batch = label.size(0)
                exo_acc.updata(exo_cls_acc, cur_batch)
                ego_acc.updata(ego_cls_acc, cur_batch)
                if global_iter_num % log_step_interval == 0:
                    logger.add_scalar("train loss", loss.item() ,global_step=global_iter_num)
                    logger.add_scalar("cls loss", cls_loss.item() ,global_step=global_iter_num)
                    logger.add_scalar("exo cls loss",exo_cls_loss.item(),global_step=global_iter_num)
                    logger.add_scalar("ego cls loss",ego_cls_loss.item(),global_step=global_iter_num)
                    logger.add_scalar("dist_loss",dist_loss.item(),global_step=global_iter_num)
                    logger.add_scalar("exo acc", exo_acc.avg, global_step=global_iter_num)
                    logger.add_scalar("ego acc", ego_acc.avg, global_step=global_iter_num)

                loss = loss.cuda()
                cls_loss = cls_loss.cuda()
                dist_loss = dist_loss.cuda()

                loss.backward()
                optimizer.step()
        
                epoch_loss += loss.item()
                

                if (step + 1) % args.show_step == 0:
                    print(
                        '{} \t Epoch:[{}/{}]\tstep:[{}/{}] \t cls_loss: {:.3f}\t dist_loss: {:.3f}\t exo_acc: {:.2f}% \t ego_acc: {:.2f}%'.format(
                            args.phase, epoch + 1, args.epochs, step + 1, len(MyDataLoader), cls_loss.item(),
                            dist_loss.item() ,exo_cls_acc, ego_cls_acc
                        ))
            KLs=[]
            SIMs=[]
            NSSs=[]
            model.eval()

            masks = torch.load(args.divide+"_gt.t7")
            for step, (image, label, mask_path) in enumerate(MytestDataLoader):
                output = model.test_forward(image.cuda())
                cam = model.get_fused_cam(target=label.long().cuda())[0].data.cpu()
                cam = np.array(cam)
                names = mask_path[0].split("/")
                key = names[-3] + "_" + names[-2] + "_" + names[-1]
                mask = masks[key]
                mask = mask / 255.0
        
                mask=cv2.resize(mask,(args.crop_size,args.crop_size))
                cam = normalize_map(cam)
                kld = cal_kl(cam, mask)
                sim=cal_sim(cam,mask)
                nss=cal_nss(cam,mask)

                KLs.append(kld)
                SIMs.append(sim)
                NSSs.append(nss)
            mKLD=sum(KLs)/len(KLs)
            mSIM=sum(SIMs)/len(SIMs)
            mNSS=sum(NSSs)/len(NSSs)
            
            print("epoch="+str(epoch)+" "+"mKLD = "+str(mKLD))
            print("epoch="+str(epoch)+" "+"mSIM = "+str(mSIM))
            print("epoch="+str(epoch)+" "+"mNSS = "+str(mNSS))
            str_2="epoch="+str(epoch)+" "+"mKLD = "+str(mKLD)+"\n"+ \
                    "epoch="+str(epoch)+" "+"mSIM = "+str(mSIM)+"\n"+\
                    "epoch="+str(epoch)+" "+"mNSS = "+str(mNSS)+"\n"+"\n"

            with open(os.path.join(args.save_path,"result.txt"),"a") as f:
                f.write(str_2)
            f.close()
            if mKLD<best_kld:
                best_kld=mKLD
                torch.save(model.state_dict(),os.path.join(args.save_path, 'best_model.pth.tar'))
            


