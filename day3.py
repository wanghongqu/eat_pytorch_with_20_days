import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import Sequential
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

'''
如果服务器具有多个GPU，tensor.cuda()方法会将tensor保存到第一块GPU上，造价于tensor.cuda(0)。此时，如果想使用第二块GPU，需手动指定tensor.cuda（1）,而这需要修改大量代码很烦琐。这里有两种替代方法：
1. 一种方法是先调用t.cuda.set_device(1)指定第二块GPU，后续的.cuda()都无须更改，切换GPU只需修改这一行代码
2. 另一种方法是设置环境变量CUDA_VISIBLE_DEVICES

支持CPU到GPU转换的包括Variable、Model、部分loss
'''

'''
持久化。在Pytorch中，以下对象可以持久化到硬盘，并能通过相应的方法加载到内存中。Tensor、Variable、nn.Module、Optimizer。本质上，这些信息最终都是保存成Tensor。Tensor的保存和加载十分简单，使用t.save和
t.load即可完成。对Module和Optimizer对象，这里建议保存对应的state_dict，而不是直接保存整个Module/Optimizer对象。Optimizer对象保存的是参数及却是信息，通过加载之前的却是信息，能够有效的减少模型震荡

'''

'''
pytorch 中是channel_first，而tensorflow中是channel last
'''
trans = transforms.Compose([
    transforms.ToTensor()
])

train_data = DataLoader(datasets.MNIST('./', train=True, transform=trans, download=True), batch_size=64, shuffle=True)
test_data = DataLoader(datasets.MNIST('./', train=False, transform=trans, download=True), batch_size=64, shuffle=True)


def build_block(in_channels, out_channels, kernel_size, stride, padding, bn=True, relu=True):
    base_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
    )
    if bn:
        base_block.add_module('bn', nn.BatchNorm2d(out_channels))
    if relu:
        base_block.add_module('relu', nn.ReLU())
    return base_block


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([
            build_block(1, 32, 3, 1, 1),
            build_block(32, 32, 3, 2, 1),

            build_block(32, 64, 3, 1, 1),
            build_block(64, 64, 3, 2, 1),

            build_block(64, 128, 3, 1, 1),
            build_block(128, 128, 3, 2, 1),

            nn.AdaptiveAvgPool2d(1),
            # nn.Linear(128, 10),
        ])
        self.head = nn.Linear(128, 10)
        pass

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        x = self.head(x)
        return x


epoch = 10000
model = Net()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()


def acc(pred, label):
    pred = torch.argmax(pred, dim=-1).type(torch.int32)
    label = label.type(torch.int32)
    return torch.mean((pred == label).type(torch.float32)).item()


for i in range(epoch):
    print(epoch, ' start...')
    for img, label in train_data:
        optim.zero_grad()
        pred = model(Variable(img))
        loss_val = loss(pred, Variable(label))
        loss_val.backward()
        acc_val = acc(pred, label)
        optim.step()
        print('loss:',loss_val.item(),'  accuracy:',acc_val)
