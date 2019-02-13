import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import math
from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


# class VGG(nn.Module):
#     # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
#     # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3, 
#     # the number of channels and hidden units are decreased compared to the architecture in paper
#     def __init__(self):
#         super(VGG, self).__init__()
#         nf = 2
#         self.res = [None for i in range(4)]
#         self.maxp = [None for i in range(4)]
#         self.pre = nn.Sequential(
#             # Stage 1
#             # TODO: convolutional layer, input channels 3, output channels 8, filter size 3
#             nn.Conv2d(3, nf, 3, stride=1, padding=1),
#             nn.BatchNorm2d(nf),
#             nn.ReLU(),
#             # TODO: max-pooling layer, size 2
#             nn.MaxPool2d(2, padding=0),
#         )
#         self.res[0] = nn.Sequential(
#             # Stage 2
#             # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
#             nn.Conv2d(nf, nf, 3, stride=1, padding=1),
#             nn.BatchNorm2d(nf),
#             nn.ReLU(),
#             nn.Conv2d(nf, nf, 3, stride=1, padding=1),
#             nn.BatchNorm2d(nf),
#         )

#         self.maxp[0] = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(2, padding=0),
#             nn.Conv2d(nf, int(2*nf), 3, stride=1, padding=1),
#         )

#         self.res[1] = nn.Sequential(
#             # Stage 2
#             # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
#             nn.Conv2d(int(2*nf), int(2*nf), 3, stride=1, padding=1),
#             nn.BatchNorm2d(int(2*nf)),
#             nn.ReLU(),
#             nn.Conv2d(int(2*nf), int(2*nf), 3, stride=1, padding=1),
#             nn.BatchNorm2d(int(2*nf)),
#         )

#         self.maxp[1] = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(2, padding=0),
#             nn.Conv2d(int(2*nf), int(4*nf), 3, stride=1, padding=1),
#         )

#         self.res[2] = nn.Sequential(
#             # Stage 2
#             # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
#             nn.Conv2d(int(4*nf), int(4*nf), 3, stride=1, padding=1),
#             nn.BatchNorm2d(int(4*nf)),
#             nn.ReLU(),
#             nn.Conv2d(int(4*nf), int(4*nf), 3, stride=1, padding=1),
#             nn.BatchNorm2d(int(4*nf)),
#         )

#         self.maxp[2] = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(2, padding=0),
#             nn.Conv2d(int(4*nf), int(8*nf), 3, stride=1, padding=1),
#         )

#         self.res[3] = nn.Sequential(
#             # Stage 2
#             # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
#             nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
#             nn.BatchNorm2d(int(8*nf)),
#             nn.ReLU(),
#             nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
#             nn.BatchNorm2d(int(8*nf)),
#         )

#         self.maxp[3] = nn.Sequential(
#             nn.ReLU(),
#             nn.MaxPool2d(2, padding=0),
#         )

#         # self.res5 = nn.Sequential(
#         #     # Stage 2
#         #     # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
#         #     nn.Conv2d(nf, nf, 3, stride=1, padding=1),
#         #     nn.ReLU()
#         #     nn.Conv2d(nf, nf, 3, stride=1, padding=1),
#         # )

#         # self.maxp5 = nn.Sequential(
#         #     nn.ReLU()
#         #     nn.MaxPool2d(2, padding=0),
#         # )

#         #     # Stage 3
#         #     # TODO: convolutional layer, input channels 16, output channels 32, filter size 3
#         #     nn.Conv2d(int(2*nf), int(4*nf), 3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     # TODO: convolutional layer, input channels 32, output channels 32, filter size 3
#         #     nn.Conv2d(int(4*nf), int(4*nf), 3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     # TODO: max-pooling layer, size 2
#         #     nn.MaxPool2d(2, padding=0),
#         #     # Stage 4
#         #     # TODO: convolutional layer, input channels 32, output channels 64, filter size 3
#         #     nn.Conv2d(int(4*nf), int(8*nf), 3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
#         #     nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     # TODO: max-pooling layer, size 2
#         #     nn.MaxPool2d(2, padding=0),
#         #     # Stage 5
#         #     # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
#         #     nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
#         #     nn.Conv2d(int(8*nf), int(8*nf), 3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     # TODO: max-pooling layer, size 2
#         #     nn.MaxPool2d(2, padding=0)
#         # )
#         self.fc = nn.Sequential(
#             # TODO: fully-connected layer (64->64)
#             nn.Linear(int(8*nf), int(8*nf), bias=True),
#             nn.ReLU(),
#             # TODO: fully-connected layer (64->10)
#             nn.Linear(int(8*nf), 10, bias=True)
#         )
#         self.nf = int(8*nf)

#     def forward(self, x):
#         x = self.pre(x)
#         for i in range(4):
#             x = self.res[i](x) + x
#             x = self.maxp[i](x)
#         # x = self.conv(x)
#         x = x.view(-1, self.nf)
#         x = self.fc(x)
#         return x


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
    net = resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)
    

if __name__== "__main__":
    main()
   
