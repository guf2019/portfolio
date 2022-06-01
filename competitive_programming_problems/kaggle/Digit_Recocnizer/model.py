import pandas
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Prepare Dataset
# load data
train = pd.read_csv(r"./mnist_kaggle/train.csv",dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%.
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42)

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,
                                           out_channels=16,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                             out_channels=32,
                                             kernel_size=5,
                                             stride=1,
                                             padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.out = nn.Linear(32*7*7, 10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x = x.view(x.size(0), -1)
        output=self.out(x)
        return output,x

net=CNN()
print(net)
lr=0.01
batch=200
epochs=6

optimizer=optim.Adam(net.parameters(), lr=lr)
loss_func=nn.CrossEntropyLoss()
train_loader = DataLoader(train, batch_size = batch, shuffle = True)
test_loader = DataLoader(test, batch_size = batch, shuffle = True)


net.train()
total_step=len(train_loader)
for epoch in range(epochs):
    if epoch + 1 % 3 == 0:
        lr = lr / 10
    for batchid,(data,target) in enumerate(train_loader):
        data, target=Variable(data.view(batch,1,28,28)),Variable(target)
        output=net(data)[0]
        loss=loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batchid %10==0:
            print("Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,batchid*len(data),len(train_loader.dataset),
                                                                           100. * batchid/len(train_loader),loss.data.item()))
#
net.eval()
with torch.no_grad():
    correct=0
    total=0
    for images, targets in test_loader:
        test_output,last_layer=net(images.view(batch,1,28,28))
        pred_y=torch.max(test_output,1)[1].data.squeeze()
        accuracy=(pred_y==targets).sum().item() /float(targets.size(0))

print("Accuracy={:.2f}".format(accuracy))

# load data
test = pd.read_csv(r"./mnist_kaggle/test.csv",dtype = np.float32)

features_numpy = test.values/255

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTest = torch.from_numpy(features_numpy)
with open('./mnist_kaggle/submit.csv', 'w') as f :
    f.write('ImageId,Label\n')

    net.eval()
    image_id = 1
    with torch.no_grad():
        for image in featuresTest:
            test_output,last_layer=net(image.view(1, 1, 28,28))
            pred_y=torch.max(test_output,1)[1].data.squeeze()
            f.write(str(image_id) + ',' + str(pred_y.item()) + '\n')
            image_id += 1
f.close()