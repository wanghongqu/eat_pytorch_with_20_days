import torch
import torch.nn as nn
import torch.nn.functional as F
'''Tensor，每个tensor中均包含requires_grad参数，来表征反向传播时，是否要计算损失对该tensor的梯度。要停止tensor梯度记录的更新，可以使用detach函数。同时，还可以使用with torch.no_grad()
来对作用域内所有required_grad=True张量的追踪
'''

class Net(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass