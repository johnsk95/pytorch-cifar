'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from models.colornet import ColorNet
from models.autoencoder_cr import MergeAutoencoder
from models.resnet2 import ResNet18


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

colorization = torch.load("checkpoints/colorization.pth")
jigsaw = torch.load("checkpoints/jigsaw_v2.pth")
rotation = torch.load("checkpoints/rotation.pth")
simsiam = torch.load("checkpoints/simsiam.pth")
reconstruction = torch.load('./checkpoints/mergeColorRotation_autoencoder_500epochs.pth')

colorization_resnet = ColorNet()
jigsaw_resnet = ResNet18()
rotation_resnet = ResNet18()
simsiam_resnet = ResNet18()

# colorization_resnet.load_state_dict(colorization['net'], strict=False)

# pretrained_jigsaw_dict = {key.replace("module.", ""): value for key, value in jigsaw['net'].items()}
# jigsaw_resnet.load_state_dict(pretrained_jigsaw_dict, strict=False)

# pretrained_rotation_dict = {key.replace("module.", ""): value for key, value in rotation['net'].items()}
# rotation_resnet.linear = nn.Linear(rotation_resnet.linear.in_features, 4)
# rotation_resnet.load_state_dict(pretrained_rotation_dict, strict=False)
# rotation_resnet.linear = nn.Linear(rotation_resnet.linear.in_features, 10)

# pretrained_simsiam_dict = {key.replace("backbone.", ""): value for key, value in simsiam['state_dict'].items()}
# simsiam_resnet.load_state_dict(pretrained_simsiam_dict, strict=False)


net = MergeAutoencoder(colorization_resnet, rotation_resnet)

net.colorization_resnet.load_state_dict(colorization['net'], strict=False)
# pretrained_jigsaw_dict = {key.replace("module.", ""): value for key, value in jigsaw['net'].items()}
# net.jigsaw_resnet.load_state_dict(pretrained_jigsaw_dict, strict=False)
pretrained_rotation_dict = {key.replace("module.", ""): value for key, value in rotation['net'].items()}
net.rotation_resnet.load_state_dict(pretrained_rotation_dict, strict=False)
# pretrained_simsiam_dict = {key.replace("backbone.", ""): value for key, value in simsiam['state_dict'].items()}
# net.simsiam_resnet.load_state_dict(pretrained_simsiam_dict, strict=False)

# freeze colorization feature extractor
for param in net.colorization_resnet.parameters():
    param.requires_grad = False
# freeze jigsaw feature extractor
# for param in net.jigsaw_resnet.parameters():
#     param.requires_grad = False
# # freeze rotation feature extractor
for param in net.rotation_resnet.parameters():
    param.requires_grad = False
# # freeze simsiam feature extractor
# for param in net.simsiam_resnet.parameters():
#     param.requires_grad = False

net.load_state_dict(reconstruction, strict=False)
# print(net.load_state_dict(reconstruction, strict=False))
net = net.to(device)

# checkpoint = torch.load('./checkpoints/merge_autoencoder_500epochs.pth')
# pretrained_jigsaw_dict = {key.replace("module.", ""): value for key, value in checkpoint['net'].items()}
# pretrained_rotation_dict = {key.replace("module.", ""): value for key, value in checkpoint['net'].items()}

# pretrained_simsiam_dict = {key.replace("backbone.", ""): value for key, value in checkpoint['state_dict'].items()}
# net.load_state_dict(pretrained_jigsaw_dict, strict=False)
# net.linear = nn.Linear(512,100)

# removed = list(net.encoder.children())[:-1]
# net.encoder= torch.nn.Sequential(*removed)

# layers = [module for module in net.encoder.modules() if not isinstance(module, nn.Sequential)]
# layers.append(nn.Linear(512,100))
# k = nn.Sequential(*layers)
# net.encoder = k


# print(net)
# x = torch.randn(1, 3, 32, 32).to(device)
# y = net(x)
# print(y.size())

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160])


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    with open('./best_ablation_cr.txt','a') as f:
        f.write(str(acc)+':'+str(epoch)+'\n')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ablation_cr.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()