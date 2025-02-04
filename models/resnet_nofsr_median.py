import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

import torch
import torch.nn as nn

class MedianPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MedianPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x = x.contiguous().view(x.size()[:4] + (-1,))
        x, _ = torch.median(x, dim=-1)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_size=(32, 32), dx=0, dy=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.image_size = image_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool_srt = nn.AvgPool2d(8)        
        # SR parameters
        self.dx = dx
        self.dy = dy

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_eval=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        feat_list = []
        # aug_list.append(x)
        # store dictionary

        b, _, h, w = x.shape

        ind_i_j = {}
        ind = 0
        for i in range(- self.dy, self.dy + 1):
            for j in range(- self.dx, self.dx + 1):
                x_sr = transforms.functional.affine(x, translate=[i,j], angle=0, scale=1, shear=0)
                out = F.relu(self.bn1(self.conv1(x_sr)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                feature_field = self.layer4(out)
                feature_field_full = torch.nn.functional.interpolate(feature_field[0:b, :, :, :], scale_factor=8, mode='nearest')
                feat_list.append(transforms.functional.affine(feature_field_full, translate=[-i,-j], angle=0, scale=1, shear=0))
                
        feat_batch = torch.stack(feat_list, dim = 0)

        feat_median, _ = torch.nanmedian(feat_batch, dim=0, keepdim=False)

        out = self.avgpool_srt(feat_median)

        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, r_outputs, nr_outputs, rec_outputs

def ResNet18_NoFSR_SR(num_classes=10, image_size=(32, 32), dx=3, dy=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, image_size=image_size, dx=dx, dy=dy)