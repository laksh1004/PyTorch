import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

x_values = [i for i in range(11)]
X_train = np.array(x_values, dtype = np.float32)
X_train.shape

X_train = X_train.reshape(-1, 1)
X_train.shape

#y=2x+1
y_values = [2*i+1 for i in x_values]
y_train = np.array(y_values, dtype = np.float32)
y_train.shape

y_train = y_train.reshape(-1, 1)
y_train.shape

class LenRegModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LenRegModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
    
input_dim = 1
output_dim = 1


model = LenRegModel(input_dim, output_dim)

model.cuda()

criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

epochs = 100
for epoch in range(epochs):
    epoch += 1
    
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(X_train).cuda())
        
    if torch.cuda.is_available():
        labels = Variable(torch.from_numpy(y_train).cuda())
    
#    inputs = Variable(torch.from_numpy(X_train))
#    labels = Variable(torch.from_numpy(y_train))
#    
    optimizer.zero_grad()
    
    outputs = model(inputs)
    
    loss = criterion(outputs, labels)
    
    loss.backward()
    
    optimizer.step()
    
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))
    

predicted = model(Variable(torch.from_numpy(X_train).cuda())).data   
#predicted = model(Variable(torch.from_numpy(X_train).cuda())).data.numpy()
predicted


save_model = False
if save_model is True:
    torch.save(model.state_dict(), 'weights.pkl')
    
load_model = False
if load_model is True:
    model.load_state_dict(torch.load('weights.pkl'))
    














