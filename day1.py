import torch
import torch.optim as optim
import torch.nn as nn
import tensorflow as tf

'''
nn.Module是pytorch中所有层、激活函数及损失的父类。
'''

'''自定义网络层
1. 自定义层时必须要继承nn.Module，并且在其构造函数中需要调用nn.Module的构造函数
2. 在该类的构造函数中必须定义模型参数，nn.Parameter
3. forward函数实现前向传播过程，其输入可以是一个或多个variable，对x的任何操作也必须是variable支持的操作
4. Module中的可学习参数可以通过named_parameters()或parameters()返回模型参数迭代器
5. 每一个Module中可以包含不同的子Module
'''


class Linear(nn.Module):
    def __init__(self, input_features, output_features):
        nn.Module.__init__(self)
        self.w = nn.Parameter(torch.rand(size=(input_features, output_features)))
        self.b = nn.Parameter(torch.rand(size=(output_features, 1)))
        self.layer = nn.Conv2d(32,32,3,2,1)
    def forward(self, input):
        return torch.sigmoid(input @ self.w + self.b)


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer1 = Linear(28 * 28, 72)
        self.layer2 = Linear(72, 10)
        pass

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        return x


net = Net()
'''
module 中parameter的全局命名规范如下。
1. Parameter直接命名。例如self.param_name = nn.Parameter(t.randn(3,4))，命名为param_name
2. 子module中的parameter，会在其名字之前加上当前module的名字。例如self.sub_module = SubModel()，
   SubModel中有个parameter的名字叫做param_name，那么二者拼接而成的parameter name 就是sub_module.param_name
'''
for name, val in net.named_parameters():
    # print(name)
    pass
'''
模型定义时，一层接一层传递麻烦。因此，ModuleList和Sequential，其中Sequential是一个特殊的Module，前向传播时会一层接一层的传递下去。ModuleList也是一个特殊的Module,可以包含几个子module。
'''
net1 = nn.Sequential(
    nn.Conv2d(3, 32, 3, 2, 1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, 2, 1),
    nn.BatchNorm2d(32),
    nn.ReLU()
)

net = nn.ModuleList([
    nn.Conv2d(3, 32, 3, 2, 1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, 2, 1),
    nn.BatchNorm2d(32),
    nn.ReLU()
])
# print(net)

'''
调整模型学习率：lr。
'''
optim = optim.Adam(net.parameters())
for group in optim.param_groups:
    group['lr'] = 0.1 * group['lr']

'''
模型初始化使用nn.init模块
'''
for i,layer in enumerate(net1.modules()):
    # print(i,'_',layer)
    if isinstance(layer,nn.Conv2d):
        nn.init.uniform(layer.weight)
        nn.init.zeros_(layer.bias)
'''
nn.Module深入分析
def __init__(self):
    self._parameters = OrderedDict() # 所有定义的self.param_name = nn.Parameters()均包含在内
    self._modules = OrderedDict() # self.layers = nn.Conv2d()等会被包含在内
    self._buffers = OrderedDict()
    self._backward_hooks = OrderedDict()
    self._forward_hooks = OrderedDict()
    self.training = True # 用于标志训练与否
'''

linear = Linear(32,64)
'''
可以通过named_modules和modules来访问每一个模型中的各个模块及子模块。也可以通过"。"号来该问各个模型的组成成分。在Python中有两个常用的内置方法：getattr和
setattr。getattr(obj,'attr1')等价于obj.attr，如果getattr函数无法找到该属性，就交给__getattr__函数处理；如果这个对象没有实现__getattr__（）方法，则会抛出异常。
result = obj.name会调用getattr(obj,'name')，如果该属性找不到，会调用obj.__getattr__('name')
obj.name = value 会调用内置的setattr(obj,'name',value),如果obj对象实现了__setattr__方法，setattr会直接调用obj.__setattr__('name',value)。nn.Module中
实现了自定义的__setattr__函数，当执行module.name=value时，会在__setattr__中判断value是否为Parameter或nn.Module对象，如果是则将这些对象加到_paramters和_modules
两个字典中；如果是其他类型的对象，如Variable、list、dict等，则会调用默认的操作，将这个值保存到__dict__中。
    
'''
# print(linear._modules)

# print(linear.b.data)
# print(linear.layer.weight)

