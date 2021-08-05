import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='dataset/mnist', train=True, transform=transform,download=True)
test_dataset = datasets.MNIST(root='dataset/mnist', train=False, transform=transform,download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        
        self.branch2_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # self.branch4_1 = 
        self.branch4_1 = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1_1(x)

        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)

        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)
        branch3 = self.branch3_3(branch3)

        branch4 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch4 = self.branch4_1(branch4)

        outputs = [branch1,branch2,branch3,branch4]
        return torch.cat(outputs, dim=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

epoch_list = []
loss_list = []
def train(epoch):
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader, 0):
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        
    epoch_list.append(epoch+1)
    loss_list.append(running_loss/900)
    print('[%d] loss: %.3f' % (epoch+1, running_loss/900))

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d  %%' % (100*correct/total))

for epoch in range(10):
    train(epoch)
    test()

plt.plot(epoch_list, loss_list)
plt.show()
