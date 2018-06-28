import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets


train_ds = dsets.MNIST(root = './data',
                       train = True,
                       transform = transforms.ToTensor(),
                       download = True)

import matplotlib.pyplot as plt
import numpy as np

#train_ds[0][0].shape
#
#show_img = train_ds[0][0].numpy().reshape(28, 28)
#plt.imshow(show_img, cmap='gray')
#train_ds[0][1]


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
test_loader = torch.utils.data.DataLoader(test_ds, batch_size, shuffle = False)

class feedforwardnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(feedforwardnn, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out
    
input_dim = 28*28
hidden_dim = 100
output_dim = 10
        
model = feedforwardnn(input_dim, hidden_dim, output_dim)

model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28).cuda())
        labels = Variable(labels.cuda())
        
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
                images = Variable(images.view(-1, 28*28).cuda())
                
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                
                correct += (predicted.cpu() == labels.cpu()).sum()
                
            accuracy = 100 * correct / total
            
            print('Iteration: {}, Loss: {}, Accuracy: {}'.format(iter, loss.data[0], accuracy))
                









