import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets


train_ds = dsets.MNIST(root = './data',
                       train = True,
                       transform = transforms.ToTensor(),
                       download = False)

import matplotlib.pyplot as plt
import numpy as np

train_ds[0][0].shape

show_img = train_ds[0][0].numpy().reshape(28, 28)
plt.imshow(show_img, cmap='gray')
train_ds[0][1]


test_ds = dsets.MNIST(root = './data',
                      train = False,
                      transform = transforms.ToTensor())

len(test_ds)

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_ds) / batch_size)
num_epochs = int(num_epochs)
num_epochs

train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle = True)
import collections
isinstance(train_loader, collections.Iterable)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size, shuffle = False)


class LogisticReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticReg, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 28*28
output_dim = 10

model = LogisticReg(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

print(model.parameters())
a = list(model.parameters())

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        iter += 1
        
        if iter % 500 == 0:
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                images = Variable(images.view(-1,28*28))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))

            


















        



