import os
import argparse
import torch
import torch.nn as nn
from models.model import MODEL

from torch.autograd import Variable
from utils.accuracy import *
from utils.lr import *
import random
import cv2
import torch.nn.functional as F
from utils.evaluation import cal_kl,cal_sim,cal_nss
import time
##  set seed
seed =0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='dataset/AGD20K')
parser.add_argument('--save_root', type=str, default='save_models')
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
parser.add_argument("--D",type=int,default=512)
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
time_str=time.strftime('%Y%m%d_%H%I%M', time.localtime(time.time()))
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
        model = MODEL(args, num_classes=args.num_classes, 
                      align_w=0.5, distil_w=0.5,
                      cls_w=1, pretrained=True,n=3,D=args.D).cuda()
        
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
        print('Train begining!')
        for epoch in range(0, args.epochs):
            model.train()
            poly_lr_scheduler(optimizer, epoch,decay_epoch=80)
            ##  accuracy
            exo_acc = AverageMeter()
            ego_acc = AverageMeter()
            epoch_loss = 0
           # optimizer.zero_grad()
            aveGrad = 0
            for step, (exocentric_image, egocentric_image, label) in enumerate(MyDataLoader):
                label = label.cuda(non_blocking=True)
                egocentric_image =  egocentric_image.cuda(
                    non_blocking=True)
                ##  backward
                optimizer.zero_grad()
                exocentric_images = Variable(torch.stack(exocentric_image, dim=1)).cuda(non_blocking=True)

                exo_out, ego_out, exo_branch, ego_branch = model(exocentric_images, egocentric_image)
                label = label.long()
                loss, cls_loss, dist_loss = model.get_loss(label,exo_branch,ego_branch,T=args.T)

                loss = loss.cuda()
                cls_loss = cls_loss.cuda()
                dist_loss = dist_loss.cuda()

                loss.backward()
                optimizer.step()
                #
                ##  count_accuracy
                cur_batch = label.size(0)
                epoch_loss += loss.item()
                exo_cls_acc = 100. * compute_cls_acc(exo_out, label)
                ego_cls_acc = 100. * compute_cls_acc(ego_out, label)
                exo_acc.updata(exo_cls_acc, cur_batch)
                ego_acc.updata(ego_cls_acc, cur_batch)

                if (step + 1) % args.show_step == 0:
                    print(
                        '{} \t Epoch:[{}/{}]\tstep:[{}/{}] \t cls_loss: {:.3f}\t dist_loss: {:.3f}\t exo_acc: {:.2f}% \t ego_acc: {:.2f}%'.format(
                            args.phase, epoch + 1, args.epochs, step + 1, len(MyDataLoader), cls_loss.item(),
                            dist_loss.item() ,exo_acc.avg, ego_acc.avg
                        ))
            KLs=[]
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
               
                KLs.append(kld)
            mKLD=sum(KLs)/len(KLs)
            print("epoch="+str(epoch)+" "+"mKLD = "+str(mKLD))
            if mKLD<best_kld:
                best_kld=mKLD
                
                torch.save(model.state_dict(),os.path.join(args.save_path, 'best_model.pth.tar'))


