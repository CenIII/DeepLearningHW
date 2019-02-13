import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import math
from collections import OrderedDict


class VGG(nn.Module):
    # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3, 
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(VGG, self).__init__()
        nf = 32
        self.res = [None for i in range(4)]
        self.maxp = [None for i in range(4)]
        self.pre = nn.Sequential(
            # Stage 1
            # TODO: convolutional layer, input channels 3, output channels 8, filter size 3
            nn.Conv2d(3, nf, 3, stride=1, padding=1),
            nn.ReLU(),
            # TODO: max-pooling layer, size 2
            nn.MaxPool2d(2, padding=0),
        )
        self.res[0] = nn.Sequential(
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
        )

        self.maxp[0] = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(nf, int(2*nf), 3, stride=1, padding=1),
        )

        self.res[1] = nn.Sequential(
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            nn.Conv2d(int(2*nf), int(2*nf), 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(2*nf), int(2*nf), 3, stride=1, padding=1),
        )

        self.maxp[1] = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(int(2*nf), int(4*nf), 3, stride=1, padding=1),
        )

        self.res[2] = nn.Sequential(
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            nn.Conv2d(int(4*nf), int(4*nf), 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(4*nf), int(4*nf), 3, stride=1, padding=1),
        )

        self.maxp[2] = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(int(4*nf), int(8*nf), 3, stride=1, padding=1),
        )

        self.res[3] = nn.Sequential(
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
        )

        self.maxp[3] = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(int(8*nf), int(16*nf), 3, stride=1, padding=1),
        )

        # self.res5 = nn.Sequential(
        #     # Stage 2
        #     # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
        #     nn.Conv2d(nf, nf, 3, stride=1, padding=1),
        #     nn.ReLU()
        #     nn.Conv2d(nf, nf, 3, stride=1, padding=1),
        # )

        # self.maxp5 = nn.Sequential(
        #     nn.ReLU()
        #     nn.MaxPool2d(2, padding=0),
        # )

        #     # Stage 3
        #     # TODO: convolutional layer, input channels 16, output channels 32, filter size 3
        #     nn.Conv2d(int(2*nf), int(4*nf), 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     # TODO: convolutional layer, input channels 32, output channels 32, filter size 3
        #     nn.Conv2d(int(4*nf), int(4*nf), 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     # TODO: max-pooling layer, size 2
        #     nn.MaxPool2d(2, padding=0),
        #     # Stage 4
        #     # TODO: convolutional layer, input channels 32, output channels 64, filter size 3
        #     nn.Conv2d(int(4*nf), int(8*nf), 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
        #     nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     # TODO: max-pooling layer, size 2
        #     nn.MaxPool2d(2, padding=0),
        #     # Stage 5
        #     # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
        #     nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
        #     nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     # TODO: max-pooling layer, size 2
        #     nn.MaxPool2d(2, padding=0)
        # )
        self.fc = nn.Sequential(
            # TODO: fully-connected layer (64->64)
            nn.Linear(int(16*nf), int(16*nf), bias=True),
            nn.ReLU(),
            # TODO: fully-connected layer (64->10)
            nn.Linear(int(16*nf), 10, bias=True)
        )
        self.nf = int(16*nf)

    def forward(self, x):
        x = self.pre(x)
        for i in range(4):
            x = self.res[i](x) + x
            x = self.maxp[i](x)
        # x = self.conv(x)
        x = x.view(-1, self.nf)
        x = self.fc(x)
        return x


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            optimizer.zero_grad()
            # TODO: forward pass
            output = net(images)
            loss = criterion(output,labels)
            # TODO: backward pass
            loss.backward()
            # TODO: optimize the network
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)
    net = VGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)
    

if __name__== "__main__":
    main()
   
