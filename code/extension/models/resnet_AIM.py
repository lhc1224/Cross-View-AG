import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util import remove_layer
from util import replace_layer
from torch.nn import functional as F
from torchvision.models.resnet import resnet50
# from torchvision.models.utils import load_state_dict_from_url
from AIM.modules.burger import HamburgerV1

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class resnet_AIM(nn.Module):
    def __init__(self, block, layers, args):
        super(resnet_AIM, self).__init__()
        self.args = args
        self.num_classes = self.args.num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.align_w = self.args.align_w
        self.distil_w = self.args.distil_w
        self.cls_w = self.args.cls_w
        self.pretrained = self.args.pretrained
        self.n = self.args.n
        self.D = self.args.D
        self.R=self.args.R
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.Hamburger = HamburgerV1(in_c=2048, n=self.n, D=self.D,R=self.R)

        ### project layer
        self.project = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(2048),
                                     nn.ReLU(inplace=True))

        self.conv6 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, self.num_classes)
        if args.init_weights:
            self._initialize_weights()

    def forward(self, exocentric, egocentric_image, label=None):
        ### ref

        b, n, c, h, w = exocentric.size()
        exocentrin_input = exocentric.view(b * n, c, h, w)
        x = self.conv1(exocentrin_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        input_x = x

        ego_x = self.conv1(egocentric_image)
        ego_x = self.bn1(ego_x)
        ego_x = self.relu(ego_x)
        ego_x = self.maxpool(ego_x)
        ego_x = self.layer1(ego_x)
        ego_x = self.layer2(ego_x)
        ego_x = self.layer3(ego_x)
        ego_x = self.layer4(ego_x)

        x, self.tmp_ego_x = self.Hamburger(input_x, ego_x)

        ### [b*n,c,h,w]
        pro_ego_x = self.project(ego_x)
        self.ego_feature_map = pro_ego_x

        exocentric_branch_conv6 = self.conv6(x)
        _, x_c, x_h, x_w = exocentric_branch_conv6.size()

        egocentric_branch = self.conv6(pro_ego_x)

        exo_pool = self.avgpool(exocentric_branch_conv6)
        exo_pool = torch.mean(exo_pool.view(-1, n, 1024), dim=1)
        self.exo_score = self.fc(exo_pool)

        # self.ego_feature_map = egocentric_branch
        ego_pool = self.avgpool(egocentric_branch)
        ego_pool = ego_pool.view(ego_pool.size(0), -1)

        ego_pool = self.fc(ego_pool)
        self.ego_score = ego_pool

        return self.exo_score, self.ego_score

    def test_forward(self, object_image):

        obj_x = self.conv1(object_image)
        obj_x = self.bn1(obj_x)
        obj_x = self.relu(obj_x)
        obj_x = self.maxpool(obj_x)
        obj_x = self.layer1(obj_x)
        obj_x = self.layer2(obj_x)
        obj_x = self.layer3(obj_x)
        obj_x = self.layer4(obj_x)

        pro_obj_x = self.project(obj_x)

        self.obj_feature_map = self.conv6(pro_obj_x)
        obj_pool = self.avgpool(self.obj_feature_map)
        obj_pool = obj_pool.view(obj_pool.size(0), -1)

        obj_pool = self.fc(obj_pool)
        self.obj_score = obj_pool

        return self.obj_score

    def get_feature_map(self, state="test"):
        if state == "train":
            return self.ref_feature_map, self.ref_score, self.obj_feature_map, self.obj_score
        else:
            return self.obj_feature_map, self.obj_score

    def get_fused_cam(self, target=None):
        
        batch, channel, _, _ = self.obj_feature_map.size()

        if target is None:
            _, target = self.score.topk(1, 1, True, True)
        # print(target)
        target = target.squeeze()

        cam_weight = self.fc.weight[target]
        cam_weight = cam_weight.view(batch, channel, 1, 1).expand_as(self.obj_feature_map)
        cam = (cam_weight * self.obj_feature_map)
        cam = cam.mean(1)

        return cam

    def get_loss(self, gt_label, T=1.0):  ## 

        ego_cls_loss = self.criterion(self.ego_score, gt_label)
        exo_cls_loss = self.criterion(self.exo_score, gt_label)

        loss_cls = self.cls_w * (exo_cls_loss + ego_cls_loss)

        self.ego_feature_map = torch.mean(F.normalize(self.ego_feature_map, dim=1), dim=1)
        self.tmp_ego_x = torch.mean(F.normalize(self.tmp_ego_x, dim=1), dim=1)
        b = self.ego_feature_map.size(0)
        self.ego_feature_map = self.ego_feature_map.view(b, -1)
        self.tmp_ego_x = self.tmp_ego_x.view(b, -1)

        loss_dist = self.distil_w * torch.mean(torch.mean((self.tmp_ego_x - self.ego_feature_map) ** 2, dim=1), dim=0)

        b, class_n = self.ego_score.size()
        ego_score_softmax = F.softmax(self.ego_score / T, dim=1).view(b, class_n, 1)
        exo_score_softmax = F.softmax(self.exo_score / T, dim=1).view(b, class_n, 1)
        ego_m = torch.bmm(ego_score_softmax, ego_score_softmax.permute(0, 2, 1))  ### [b,n_class,n_class]
        exo_m = torch.bmm(exo_score_softmax, exo_score_softmax.permute(0, 2, 1))  ### [b,n_class,n_class]
        tmp_loss = -self.align_w * T * T * torch.mean(torch.sum(torch.sum(torch.log(ego_m) * exo_m, dim=2), dim=1))

        loss_dist += tmp_loss

        loss = loss_cls + loss_dist

        return loss, loss_cls, loss_dist, exo_cls_loss, ego_cls_loss

    def do_normalize(self, heat_map):
        N, H, W = heat_map.shape

        batch_mins, _ = torch.min(heat_map.view(N, -1), dim=-1)
        batch_maxs, _ = torch.max(heat_map.view(N, -1), dim=-1)
        normalize_map = (heat_map - batch_mins) / (batch_maxs - batch_mins)
        return normalize_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )
def load_pretrained_model(model):
    strict_rule = True
    state_dict = torch.load('weights/resnet50-19c8e357.pth')

    state_dict = remove_layer(state_dict, 'fc')
    strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model

def model(args, pretrained=True):
    model = resnet_AIM(Bottleneck, [3, 4, 6, 3], args)
    if pretrained:
        load_pretrained_model(model)
    return model
