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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_size=(32, 32), dx=0, dy=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.image_size = image_size
        
        # SR parameters
        self.dx = dx
        self.dy = dy
        self.patch_size = 8
        self.patch_pixels = self.patch_size ** 2

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool_srt = nn.AvgPool2d(8)        


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def calculate_sr_weights(self, i, j):
        # get magnitude of shifts 
        h_mag = abs(i)
        w_mag = abs(j)
        
        # Calculate weights based on overlap areas
        weight_center = (self.patch_size - h_mag) * (self.patch_size - w_mag) / self.patch_pixels
        weight_h = h_mag * (self.patch_size - w_mag) / self.patch_pixels
        weight_w = (self.patch_size - h_mag) * w_mag / self.patch_pixels
        weight_corner = h_mag * w_mag / self.patch_pixels
    
        return weight_center, weight_h, weight_w, weight_corner

    def forward(self, x, is_eval):
        b, _, h, w = x.shape # b (batch_size) * c (3) * h (32) * w (32)
        
        # create list of augmented images
        aug_list = []
        ind_i_j = {}
        ind = 0
        for i in range(- self.dy, self.dy + 1):
            for j in range(- self.dx, self.dx + 1):
                aug_list.append(transforms.functional.affine(x, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1
    
        # concatenate all augmented images into a single batch
        x = torch.cat(aug_list, dim=0)
        
        # run through resnet
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feature_field = self.layer4(out)

        # init with first feature
        feature_field_ensembled = feature_field[0:b, :, :, :].clone()

        # for each shifted feature
        # for each shifted feature
        for k in range(1, ind):
            i, j = ind_i_j[k]
            h_mag = abs(i)
            w_mag = abs(j)

            feature_field_temp = feature_field[k*b:(k+1)*b, :, :, :]
            feature_field_ensembled += feature_field_temp * ((self.patch_size - h_mag) * (self.patch_size - w_mag) / self.patch_pixels)
         
            if i > 0:
                feature_field_ensembled[:,:,:-1,:] += feature_field_temp[:,:,1:,:]*(h_mag*(self.patch_size-w_mag) / self.patch_pixels)
            elif i < 0:
                feature_field_ensembled[:,:,1:,:] += feature_field_temp[:,:,:-1,:]*(h_mag*(self.patch_size-w_mag) / self.patch_pixels)

            if j > 0:
                feature_field_ensembled[:,:,:,:-1] += feature_field_temp[:,:,:,1:]*((self.patch_size-h_mag)*w_mag / self.patch_pixels)
            elif j < 0:
                feature_field_ensembled[:,:,:,1:] += feature_field_temp[:,:,:,:-1]*((self.patch_size-h_mag)*w_mag / self.patch_pixels)

            if i > 0 and j > 0:
                feature_field_ensembled[:,:,:-1,:-1] += feature_field_temp[:,:,1:,1:]*(h_mag*w_mag / self.patch_pixels)  
            elif i > 0 and j < 0:
                feature_field_ensembled[:,:,:-1,1:] += feature_field_temp[:,:,1:,:-1]*(h_mag*w_mag / self.patch_pixels)  
            elif i < 0 and j > 0:
                feature_field_ensembled[:,:,1:,:-1] += feature_field_temp[:,:,:-1,1:]*(h_mag*w_mag / self.patch_pixels)  
            elif i < 0 and j < 0:
                feature_field_ensembled[:,:,1:,1:] += feature_field_temp[:,:,:-1,:-1]*(h_mag*w_mag / self.patch_pixels)  

        # pool without interpolation
        out = nn.AdaptiveAvgPool2d(1)(feature_field_ensembled)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, [], [], []

def ResNet18_NoFSR_SR(num_classes=10, image_size=(32, 32), dx=3, dy=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, image_size=image_size, dx=dx, dy=dy)