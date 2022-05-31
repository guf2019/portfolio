import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(28*28,200)
        self.fc2=nn.Linear(200,200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

net=Net()
print(net)
lr=0.001
batch=200
epochs=10
optimizer=optim.Adam(net.parameters(), lr=lr)
loss_func=nn.NLLLoss()
train_loader=DataLoader(datasets.MNIST("../data",
                                       train=True,
                                       download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])),
                                       batch_size=batch,shuffle=True)
test_loader=DataLoader(datasets.MNIST("../data",
                                       train=False,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])),
                                       batch_size=batch,shuffle=True)

for epoch in range(epochs):
    for batchid,(data,target) in enumerate(train_loader):
        data, target=Variable(data),Variable(target)
        data=data.view(-1,28*28)
        optimizer.zero_grad()
        output=net(data)
        loss=loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batchid %10==0:
            print("train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,batchid*len(data),len(train_loader.dataset),
                                                                           100. * batchid/len(train_loader),loss.data.item()))

test_loss=0
correct=0
for data,target in test_loader:
    data,target=Variable(data,volatile=True),Variable(target)
    data = data.view(-1, 28 * 28)
    output=net(data)
    test_loss+=loss_func(output, target).item()
    pred=output.data.max(1)[1]
    correct+=pred.eq(target.data).sum()

test_loss/=len(test_loader.dataset)
print("\n test set: average loss: {:.4f}, Accuracy:{}/{} ({:.0f}%)\n".format(test_loss,correct,len(test_loader.dataset),100. * correct / len(test_loader.dataset)))