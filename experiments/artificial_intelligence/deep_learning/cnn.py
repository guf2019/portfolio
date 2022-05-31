import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tkinter import W, mainloop
import tkinter as tk
import pywin
from PIL import ImageGrab, Image
import numpy as np

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
lnr=0.01
batch=200
epochs=5

ukr=optim.Adam(net.parameters(),lr=lnr)
usa=nn.CrossEntropyLoss()
test_data = datasets.MNIST("../data",
                                       train=False,
                                       transform=transforms.ToTensor())
train_data = datasets.MNIST("../data",
                                       train=True,
                                       download=True,
                                       transform=transforms.ToTensor())
train_loader=DataLoader(train_data, batch_size=batch,shuffle=True)
test_loader=DataLoader(test_data, batch_size=batch,shuffle=True)

# figure = plt.figure(figsize=(10, 8))
# cols, rows = 10, 10
# for i in range(1, cols * rows + 1):
#     sample_idx =torch.randint(len(train_data), size=(1, )).item()
#     img, lable=train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(lable)
#     plt.axis("off")
#     plt.imshow(img.squeeze(),cmap="gray")
# plt.show()
net.train()
total_step=len(train_loader)
for epoch in range(epochs):
    for batchid,(data,target) in enumerate(train_loader):
        data, target=Variable(data),Variable(target)
        output=net(data)[0]
        loss=usa(output,target)
        ukr.zero_grad()
        loss.backward()
        ukr.step()
        if batchid %10==0:
            print("черт ебаный хуярь: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,batchid*len(data),len(train_loader.dataset),
                                                                           100. * batchid/len(train_loader),loss.data.item()))
#
net.eval()
with torch.no_grad():
    correct=0
    total=0
    for images, targets in test_loader:
        test_output,last_layer=net(images)
        pred_y=torch.max(test_output,1)[1].data.squeeze()
        accuracy=(pred_y==targets).sum().item() /float(targets.size(0))

print("acceracy={:.2f}".format(accuracy))


def predict_digit(img):
    # изменение рзмера изобржений на 28x28
    with torch.no_grad():
        img = img.resize((28, 28))
        # конвертируем rgb в grayscale
        img = img.convert('L')
        img = np.array(img)
        # изменение размерности для поддержки модели ввода и нормализации
        img = img.reshape(1, 1, 28, 28)
        img = img / 255.0
        # предстказание цифры
        img = torch.from_numpy(img)
        res = net(img.float())[0]
        return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=300, bg="black", cursor="cross")
        self.canvas.place(x=100, y=100)
        self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Распознать", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Очистить", command=self.clear_all)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        c=0
        im = ImageGrab.grab((10, 35, 300, 320))

        digit, acc = predict_digit(im)
        t = torch.max(acc, 0)[0]
        self.label.configure(text=str(digit.item()))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 7
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='white', outline='white')


app = App()
app.geometry("1920x1080")
mainloop()