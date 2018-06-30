import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())



batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        

        self.layer_dim = layer_dim
        

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        

        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):

        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        
        out, (hn, cn) = self.lstm(x, (h0,c0))
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 28
hidden_dim = 100
layer_dim = 3  
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)



model.cuda()
    

criterion = nn.CrossEntropyLoss()


learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  




seq_dim = 28  

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):


        images = Variable(images.view(-1, seq_dim, input_dim).cuda())
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
                images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                
                correct += (predicted.cpu() == labels.cpu()).sum()
                         
            accuracy = 100 * correct / total
            
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))
