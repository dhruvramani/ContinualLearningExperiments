'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.resnetlin import ResNet18
from utils import progress_bar

if not os.path.isdir('logs'):
    os.mkdir('logs')

if not os.path.isdir('embedding'):
    os.mkdir('embedding')

if not os.path.isdir('activs'):
    os.mkdir('activs')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--onlyclass', default=1, type=int, help='Only training one class')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
filepath = "./embedding/class_"

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
cpunet = net
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# Training
def train(epoch, curr_class, old_classes):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, loss_zero = 0, 0
    correct, correct_zero = 0, 0
    total, total_zero     = 0, 0

    if(len(old_classes) == 0):
        params = net.parameters()
    else :
        params = list(cpunet.layer4.parameters()) + list(cpunet.linear1.parameters()) + list(cpunet.linear2.parameters())
    
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)


'''    for old_class in old_classes:
        with open(filepath + str(old_class) + ".pkl", 'rb') as file:
            unpickler = pickle._Unpickler(file)
            unpickler.encoding = 'latin1'
            contents = unpickler.load()
            
            X, Y = np.asarray(contents['data'], dtype=np.float32), np.asarray(contents['labels'])
        X, Y = Variable(torch.from_numpy(X), requires_grad=False), Variable(torch.from_numpy(Y), requires_grad=False)
        X, Y = X.to(device), Y.to(device)
        
        optimizer.zero_grad()
        outputs, linact = net(X, old_class=True)
        linact = linact.data.cpu().numpy()
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        with open("./activs/linear_{}_{}.pkl".format(curr_class, old_class), "wb+") as file:
            pickle.dump(linact, file, protocol=pickle.HIGHEST_PROTOCOL)

        if(old_class == 0):
            loss_zero += loss.item()
            total_zero += Y.size(0)
            _, predicted = outputs.max(1)
            correct_zero += predicted.eq(Y).sum().item()

            with open("./logs/train_zero_loss.log", "a+") as lfile:
                lfile.write(str(loss_zero / total_zero))
                lfile.write("\n")

            with open("./logs/train_zero_acc.log", "a+") as afile:
                afile.write(str(correct_zero / total_zero))
                afile.write("\n")

    print("Trained for Previous Classes : {}".format(old_classes))'''

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        contents = dict()
        inputs, targets = inputs.cpu().numpy(), targets.cpu().numpy()
        idx = np.where(targets == curr_class)
        inputs, targets = inputs[idx], targets[idx]
        np_targets = targets
        inputs, targets = Variable(torch.from_numpy(inputs), requires_grad=False), Variable(torch.from_numpy(targets), requires_grad=False)
        inputs, targets = inputs.to(device), targets.to(device)

        activs, outputs, linact = net(inputs, old_class=False)
        activs = activs.data.cpu().numpy()

        contents['data'] = activs
        contents['labels'] = np_targets

        if(not os.isdir(filepath + str(curr_class))):
            os.mkdir(filepath + str(curr_class))

        with open(filepath + str(curr_class) + "/" + str(batch_idx) + ".pkl", "wb+") as file:
            pickle.dump(contents, file, protocol=pickle.HIGHEST_PROTOCOL)

        for old_class in old_classes:
            lent = len([name for name in os.listdir(filepath + str(old_classes)) if os.path.isfile(name)])
            with open(filepath + str(old_class) + "/" + str(int(batch_idx / lent)) + ".pkl", 'rb') as file:
                unpickler = pickle._Unpickler(file)
                unpickler.encoding = 'latin1'
                old_contents = unpickler.load()

            X, Y = np.asarray(old_contents['data'], dtype=np.float32), np.asarray(old_contents['labels'])
            activs = np.concatenate((activs, X))
            np_targets = np.concatenate((np_targets, Y))

        inputs, targets = Variable(torch.from_numpy(activs), requires_grad=False), Variable(torch.from_numpy(np_targets), requires_grad=False)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs, linact = net(X, old_class=True)
        linact = linact.data.cpu().numpy()
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("./logs/train_loss_{}.log".format(curr_class), "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("./logs/train_acc_{}.log".format(curr_class), "a+") as afile:
            afile.write("{}\n".format(correct / total))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, old_classes_arr, curr_class):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    classes = old_classes_arr + [curr_class]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            real_inp, real_targ = inputs, targets
            to_passinp, to_passtargets = inputs, targets
            
            for cu_class in classes:
                inputs, targets = real_inp.cpu().numpy(), real_targ.cpu().numpy()
                idx = np.where(targets == cu_class)
                inputs, targets = inputs[idx], targets[idx]
                to_passinp = np.concatenate((to_passinp, inputs))
                to_passtargets = np.concatenate((to_passtargets, targets))
            
            to_passinp, to_passtargets = Variable(torch.from_numpy(to_passinp), requires_grad=False), Variable(torch.from_numpy(to_passtargets), requires_grad=False)
            to_passinp, to_passtargets = to_passinp.to(device), to_passtargets.to(device)

            _, outputs, _ = net(inputs, old_class=False)
            loss = criterion(outputs, to_passtargets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += to_passtargets.size(0)
            correct += predicted.eq(to_passtargets).sum().item()

            with open("./logs/test_loss_{}.log".format(curr_class), "a+") as lfile:
                lfile.write(str(test_loss / total))
                lfile.write("\n")

            with open("./logs/test_acc_{}.log".format(curr_class), "a+") as afile:
                afile.write(str(correct / total))
                afile.write("\n")

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx + 1), 100. * correct/total, correct, total))

    # Save checkpoint.
    acc = 100. * correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    return best_acc


for i in range(10):
    old_classes_arr = [j for j in range(i)]
    old_acc, count = 0, 1
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, i, old_classes_arr)
        acc = test(epoch, old_classes_arr, i)
        old_acc = old_acc + (acc - old_acc) / count
        if(int(count % 10) == 0):
            if(acc - old_acc <= 1):
                break
            count = 0
            old_acc = 0
        count += 1