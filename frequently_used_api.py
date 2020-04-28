import tensorflow as tf
import torch

'''
修改tensor 维度
'''
# squeeze、unsqueeze
arr = torch.arange(36, dtype=torch.int32).reshape(3, 3, 4)
brr = torch.unsqueeze(arr, 0)
# print(brr.shape)
crr = torch.squeeze(brr)
# print(crr.shape)

# view
# print(arr.view(size=(3,12)))

'''
tensor支持与numpy类似的索引操作，索引结果与原始数据共用内存，即修改一个，另外一个也会根着改变
'''
bool_msk = torch.logical_and(arr[..., 0] % 12 == 0, arr[..., 0] > 12)
arr[bool_msk] = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
# print(arr)


'''
tensor大小比较,max函数支持以下三种操作，1. 返回一个数组中的最大元素。2.返回沿某个轴的最大元素。3.两个同维度的tensor作比较
'''
arr = torch.randn((4, 4))
print(arr)
val, _ = torch.max(arr, dim=0, keepdim=False)
# print(val)
idx = torch.argsort(arr[..., 0], descending=True)
print(idx)
print(arr[idx])

'''
广播法则，是科学计算中经常使用 的一个技巧，它在快速执行向量化的同时不会占用额外的内存。Numpy的广播法则定义如下：
1. 让所有输入数组 都向其中shape最长的数组看齐，shape中不足的部分通过在前面加1补齐
2. 两个数组要么在某一个维度一致，要么其中一个为1，否则不能计算。
手动广播在以上两步中调用的函数：
1. 当两个tensor长度不一致时，首先调用unsqueeze或view，对数据某一维填充1
2. 同一维度中元素个数不同时，调用expand或expand_as来进行复制。当然，该复制是共享内存，不会占用额外空间。
'''

'''
tensor的保存和恢复。torch.save(arr,'a.pth')
'''

'''
tensor与numpy的互转v,/'''
# arr.numpy()
# torch._copy_from()

'''
    
'''