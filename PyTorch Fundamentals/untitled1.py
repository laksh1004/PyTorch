#pytorch variables
#a variable wraps a tensor which allows accumulation of gradients
import torch
from torch.autograd import Variable

#requires_grad=True is set when we need to compute gradient on this variable
a = Variable(torch.ones(2,2), requires_grad=True)
a
b = Variable(torch.ones(2,2), requires_grad=True)
print(torch.add(a,b))




#Gradient calculation
#y = 5(x+1)^2
x = Variable(torch.ones(2), requires_grad=True)
y = 5 * ( x + 1 ) ** 2
0
#backward or gradient can be called only on scalar i.e. one element tensor
#o = 1/n(sum(y)), n = 2 
o = (1/2) * torch.sum(y)
o
o.backward()
x.grad