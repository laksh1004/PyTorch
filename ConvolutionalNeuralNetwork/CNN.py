import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets

train_ds = dsets.MNIST(root = './data',
                       train = True,
                       transform = transforms.ToTensor(),
                       download = True)

test_ds = dsets.MNIST(root = './data',
                       train = False,
                       transform = transforms.ToTensor(),
                       download = False)

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_ds) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size, shuffle = False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.cnn2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

model = CNN()
model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), learning_rate)








