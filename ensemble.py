import os
from re import M
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from models.colornet import ColorNet
from models.resnet import ResNet18
from models.simsiam import SimSiam
from utils import progress_bar

# class weightMLP(nn.Module):
#     def __init__(self, m1, m2, m3, num_tasks=3):
#         super(weightMLP, self).__init__()
#         self.softmax = nn.Softmax(dim=1)
#         # load different pretext tasks
#         self.m1 = m1 # colorization
#         self.m2 = m2 # jigsaw
#         self.m3 = m3 # rotation

#         self.m1_full = m1
#         self.m2_full = m2
#         self.m3_full = m3
#         # self.m4 = m4 # simsiam
#         # remove last fc layer
#         self.m1.linear = nn.Identity()
#         self.m2.linear = nn.Identity()
#         self.m3.linear = nn.Identity()
#         # self.m4.linear = nn.Identity()
#         # concatenated layer predicts weights for each task
#         # self.weights = nn.Linear(512*3, num_tasks)
#         self.weights = nn.Linear(512*3, 512)
#         self.weights1 = nn.Linear(512, 256)
#         self.weights2 = nn.Linear(256, 3)
#     def forward(self, x):
#         out1 = self.m1(x.clone())
#         out1 = out1.view(out1.size(0), -1)
#         out2 = self.m2(x.clone())
#         out2 = out2.view(out2.size(0), -1)
#         out3 = self.m3(x.clone())
#         out3 = out3.view(out3.size(0), -1)
#         # out4 = self.m4(x.clone())
#         # out4 = out4.view(out4.size(0), -1)
#         out = torch.cat((out1, out2, out3), dim=1)
#         out = self.weights(out)
#         out = self.weights1(out)
#         out = self.weights2(out)
#         out = self.softmax(out)

#         y1 = self.m1_full(x.clone())
#         # print('hey ', y1.size())
#         y2 = self.m2_full(x.clone())
#         y3 = self.m3_full(x.clone())
#         wavg = (out[0,0] * y1 + out[0,1] * y2 + out[0,2] * y3) / (out[0,0] + out[0,1] + out[0,2])
#         return wavg

class weightMLP(nn.Module):
    def __init__(self, m1, m2, m3, num_tasks=3):
        super(weightMLP, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        # load different pretext tasks
        self.m1 = m1 # colorization
        self.m2 = m2 # jigsaw
        self.m3 = m3 # rotation
        # self.m4 = m4 # simsiam
        # remove last fc layer
        self.m1.linear = nn.Identity()
        self.m2.linear = nn.Identity()
        self.m3.linear = nn.Identity()
        # self.m4.linear = nn.Identity()
        # concatenated layer predicts weights for each task
        # self.weights = nn.Linear(512*3, num_tasks)
        self.weights = nn.Linear(512*3, 512)
        self.weights1 = nn.Linear(512, 256)
        self.weights2 = nn.Linear(256, 100)
    def forward(self, x):
        out1 = self.m1(x.clone())
        out1 = out1.view(out1.size(0), -1)
        out2 = self.m2(x.clone())
        out2 = out2.view(out2.size(0), -1)
        out3 = self.m3(x.clone())
        out3 = out3.view(out3.size(0), -1)
        # out4 = self.m4(x.clone())
        # out4 = out4.view(out4.size(0), -1)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.weights(out)
        out = self.weights1(out)
        out = self.weights2(out)
        return out

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

colorization = torch.load("checkpoints/colorization.pth")
jigsaw = torch.load("checkpoints/jigsaw_v2.pth")
rotation = torch.load("checkpoints/rotation.pth")
simsiam = torch.load("checkpoints/simsiam.pth")

colorization_resnet = ColorNet().to(device)
jigsaw_resnet = ResNet18().to(device)
rotation_resnet = ResNet18().to(device)
simsiam_resnet = ResNet18().to(device)

colorization_resnet.load_state_dict(colorization['net'], strict=False)

pretrained_jigsaw_dict = {key.replace("module.", ""): value for key, value in jigsaw['net'].items()}
jigsaw_resnet.load_state_dict(pretrained_jigsaw_dict, strict=False)

pretrained_rotation_dict = {key.replace("module.", ""): value for key, value in rotation['net'].items()}
rotation_resnet.linear = nn.Linear(rotation_resnet.linear.in_features, 4)
rotation_resnet.load_state_dict(pretrained_rotation_dict, strict=False)
rotation_resnet.linear = nn.Linear(rotation_resnet.linear.in_features, 10)

pretrained_simsiam_dict = {key.replace("backbone.", ""): value for key, value in simsiam['state_dict'].items()}
simsiam_resnet.load_state_dict(pretrained_simsiam_dict, strict=False)

# freeze colorization feature extractor
for param in colorization_resnet.parameters():
    param.requires_grad = False
# freeze jigsaw feature extractor
for param in jigsaw_resnet.parameters():
    param.requires_grad = False
# freeze rotation feature extractor
for param in rotation_resnet.parameters():
    param.requires_grad = False
# freeze simsiam feature extractor
for param in simsiam_resnet.parameters():
    param.requires_grad = False

net = weightMLP(jigsaw_resnet, rotation_resnet, simsiam_resnet)
net = net.to(device)

# t = torch.randn(1,3,32,32).to(device)
# output = net(t)
# print('out: ', output.size()) # [1,3]

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
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001,
#                       momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    # jigsaw_resnet.train()
    # rotation_resnet.train()
    # simsiam_resnet.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs) # [1, # pretext tasks]
        # calculate weighted average
        # y1 = jigsaw_resnet(inputs)
        # y2 = rotation_resnet(inputs)
        # y3 = simsiam_resnet(inputs)
        # w1 = outputs[0,0]
        # w2 = outputs[0,1]
        # w3 = outputs[0,2]
        # waverage = (w1 * y1 + w2 * y2 + w3 * y3) / (w1 + w2 + w3)
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
    # jigsaw_resnet.eval()
    # rotation_resnet.eval()
    # simsiam_resnet.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # print('weights: ', outputs)
            # calculate weighted average
            # y1 = jigsaw_resnet(inputs)
            # y2 = rotation_resnet(inputs)
            # y3 = simsiam_resnet(inputs)
            # w1 = outputs[0,0]
            # w2 = outputs[0,1]
            # w3 = outputs[0,2]
            # waverage = (w1 * y1 + w2 * y2 + w3 * y3) / (w1 + w2 + w3)
            # loss = criterion(waverage, targets)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ensemble.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
