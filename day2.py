import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader


def my_func(x):
    pass


transforms.Compose([
    transforms.Resize(224),  # 保持长宽比例不变，将其中的短边变换成224
    transforms.CenterCrop(224),  # 切出224*224大小的图片
    transforms.ToTensor(),  # 将Image转换成Tensor，归一化至【0，1】
    transforms.Lambda(lambda x: my_func(x))
])


class MyData(Dataset):
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


dataset = MyData()
data_loader = DataLoader(dataset, 32, True, shuffle=True)

'''
torchvision是计算机识觉包，主要由三个部分组成:model、dataseet、transforms，也即内置常用的算法模型、数据集及转换操作
其中transforms中涵盖两类操作：针对tensor、针对pillow Image
'''

