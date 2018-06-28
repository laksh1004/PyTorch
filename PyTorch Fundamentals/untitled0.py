#Introduction to PyTorch
#Matrices in numpy=Ndarray, PyTorch=tensors

import numpy as np
import torch

#Simple array
arr = [[1,2], [3,4]]
print(arr)

#Ndarray
np.array(arr)
#torch tensor
torch.Tensor(arr)

#Ndarray with default values
np.ones((2,2))
torch.ones((2,2))
#torch tensor with default values
np.random.rand(2,2)
torch.rand(2,2)

#Ndarray with random values
np.random.seed(2)
np.random.rand(2,2)
#torch tensor with random values
torch.manual_seed(2)
torch.rand(2,2)





#setting seed while using gpu
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2)
torch.rand(2,2)





#numpy to torch bridge
np_array = np.ones((2,2))
torch_tensor = torch.from_numpy(np_array)
print(torch_tensor)

print(type(torch_tensor))

#this will give error because pytorch supports only the below written datatypes
np_array_new = np.ones((2,2), dtype=np.int8)
torch.from_numpy(np_array_new)

#Conversions
#Numpy  =  Torch
#int64  =  LongTensor
#int32  =  IntegerTensor
#uint8  =  ByteTensor
#float64=  DoubleTensor
#float32=  FloatTensor
#double =  DoubleTensor
np_array_new = np.ones((2,2), dtype=np.int64)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2,2), dtype=np.int32)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2,2), dtype=np.uint8)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2,2), dtype=np.float64)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2,2), dtype=np.float32)
torch.from_numpy(np_array_new)

np_array_new = np.ones((2,2), dtype=np.double)
torch.from_numpy(np_array_new)





#torch to numpy bridge
torch_tensor = torch.rand(2,2)
torch_to_numpy = torch_tensor.numpy()
print(type(torch_to_numpy))
#changing tensor type
torch_tensor = torch.DoubleTensor





#tensors on cpu vs gpu
#by default, it is on cpu
tensor_cpu = torch.ones(2,2)
#cpu to gpu
if torch.cuda.is_available():
    tensor_gpu = tensor_cpu.cuda





#mathematical operations on tensors
#resizing tensor
a = torch.ones(2,2)
print(a.size())
a.view(4)
a.view(4).size()

#element wise addition
a = torch.ones(2,2)
b = torch.ones(2,2)
c = a+b
print(c)
c = torch.add(a,b)
c
#inplace addition
#c = c+a
c.add_(a)

#element wise sub
c = a-b
c
#not inplace
print(a.sub(b))
#inplace
c.sub_(a)

#element wise multiplication
c = a*b
c
torch.mul(a,b)
#inplace
c.mul_(torch.rand(2,2))
#same in division

#tensor mean
a = torch.Tensor([1,2,3,4,5,6,7,8,9,10])
a.mean()
b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
b.mean(dim=0)
b.mean(dim=1)

#standard deviation
a.std(dim = 0)








