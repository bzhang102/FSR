import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from models.BaseModel import BaseModelDNN

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--load_name', type=str, help='specify checkpoint load name', default='cifar10_resnet18')
parser.add_argument('--train_type', choices=['AT', 'TRADES', 'MART'], default='AT')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--sr_type', choices=['base', 'full', 'single'], default='base')
parser.add_argument('--model_type', choices=['resnet', 'wideresnet'], default='resnet')
parser.add_argument('--dxdy', default=3, type=int)
parser.add_argument('--bs', default=32, type=int)
parser.add_argument('--device', type=int)

args = parser.parse_args()
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

if args.model_type == 'resnet':
    if args.sr_type == 'base':
        from models.resnet_nofsr import ResNet18_NoFSR as net
    elif args.sr_type == 'full':
        from models.resnet_nofsr_sr import ResNet18_NoFSR_SR as net
    elif args.sr_type == 'single':
        from models.resnet_nofsr_sr import ResNet18_NoFSR_SR as net 
        from models.resnet_nofsr import ResNet18_NoFSR as net_single
elif args.model_type == 'wideresnet':
    if args.sr_type == 'base':
        from models.wideresnet_nofsr import WideResNet34_NoFSR as net
    elif args.sr_type == 'full':
        from models.wideresnet_nofsr_sr import WideResNet34_NoFSR_SR as net
    elif args.sr_type == 'single':
        from models.wideresnet_nofsr_sr import WideResNet34_NoFSR_SR as net
        from models.wideresnet_nofsr import WideResNet34_NoFSR as net_single

if args.dataset == 'cifar10':
    image_size = (32, 32)
    num_classes = 10
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
elif args.dataset == 'svhn':
    image_size = (32, 32)
    num_classes = 10
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)

class CE_loss(nn.Module):
    def forward(self, logits_final, target):
        return F.cross_entropy(logits_final, target)

class CW_loss(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, logits_final, target):
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,)).to(device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        real = (target_onehot * logits_final).sum(1)
        other = ((1. - target_onehot) * logits_final - target_onehot * 10000.).max(1)[0]
        return -torch.clamp(real - other + 50, min=0.).sum()

class Classifier(BaseModelDNN):
    def __init__(self):
        super(BaseModelDNN).__init__()
        if args.sr_type == 'base':
            self.net = net(num_classes=num_classes, image_size=image_size).to(device)
            self.net_single = self.net
        elif args.sr_type == 'single':
            self.net = net(num_classes=num_classes, image_size=image_size, dx=args.dxdy, dy=args.dxdy).to(device)
            self.net_single = net_single(num_classes=num_classes, image_size=image_size).to(device)
        else:
            self.net = net(num_classes=num_classes, image_size=image_size, dx=args.dxdy, dy=args.dxdy).to(device)
            self.net_single = self.net
            
    def predict(self, x, is_eval=True):
        return self.net(x, is_eval=is_eval)
        
    def predict_single_pass(self, x, is_eval=True):
        return self.net_single(x, is_eval=is_eval)

def main():
    model = Classifier()
    model_name = 'resnet18' if args.model_type == 'resnet' else 'wrn34'
    checkpoint = torch.load(f'weights/{args.train_type}_weights/{args.dataset}/{model_name}_nofsr/{args.load_name}.pth')
    model.net.load_state_dict(checkpoint)
    model.net_single.load_state_dict(checkpoint)
    model.net.eval()
    model.net_single.eval()

    from advertorch_fsr.attacks import FGSM, LinfPGDAttack
    
    attacks = [
        (FGSM, dict(loss_fn=CE_loss(), eps=8/255, clip_min=0.0, clip_max=1.0, targeted=False), 'FGSM'),
        (LinfPGDAttack, dict(loss_fn=CE_loss(), eps=8/255, nb_iter=20, eps_iter=0.8/255, 
                            rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-20'),
        (LinfPGDAttack, dict(loss_fn=CE_loss(), eps=8/255, nb_iter=100, eps_iter=0.8/255,
                            rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-100'),
        (LinfPGDAttack, dict(loss_fn=CW_loss(num_classes=num_classes), eps=8/255, nb_iter=30, 
                            eps_iter=0.8/255, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False), 'C&W'),
    ]

    attack_results = []
    for attack_class, attack_kwargs, name in attacks:
        from metric.classification import defense_success_rate
        message, defense_success, natural_success = defense_success_rate(
            model.predict, model.predict_single_pass, testloader, 
            attack_class, attack_kwargs, device=device
        )
        print(f'{name}: {message}')
        attack_results.append(defense_success)
    
    attack_results.append(natural_success)
    attack_results = torch.cat(attack_results, 1)
    attack_results = attack_results.sum(1)
    attack_results[attack_results < len(attacks) + 1] = 0.
    
    dataset_size = 10000 if args.dataset == 'cifar10' else 26032
    print(f'Ensemble: {100. * attack_results.count_nonzero() / dataset_size:.2f}%')

if __name__ == '__main__':
    main()