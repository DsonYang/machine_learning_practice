import numpy as np
import re
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


batch_size = 64
train_raw_data_path = 'dataset/kaggle/titanic/train.csv'
train_new_data_path = 'dataset/kaggle/titanic/train_new.csv'
test_raw_data_path = 'dataset/kaggle/titanic/test.csv'
# print(train_dataset[0])

class MyDataset(Dataset):
    def __init__(self, isTest):
        self.isTest = isTest
        self.formatData()
        self.x_data = self.data[:,[0,1,2,3,4,5]]
        self.y_data = self.data[:,[6]]
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

    def formatData(self):
        arr = []
        with open(train_raw_data_path, 'r') as f:
            for index,l in enumerate(f):
                # if self.isTest and index < 890:
                #     continue
                # if not self.isTest and index > 890:
                #     continue
                l = re.sub(r',".*",',',0,',l)
                l = re.sub(r',male,',',0,',l)
                l = re.sub(r',female,',',1,',l)
                l = re.sub(r',,',',0,',l)
                l = l.split(',')
                arr.append(l)
            # print(arr)
            arr = np.delete(arr,[0,3,8,10,11],axis=1)
            # print(arr)
            arr[:,[0, 6]] = arr[:,[6, 0]]
            # print(arr)
            np.savetxt(train_new_data_path, arr, fmt='%s')
            arr = np.delete(arr,[0],axis=0)
            arr = np.array(arr, dtype=np.float32)
            self.data = arr


train_data = MyDataset(0)
test_data = MyDataset(1)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(6,4)
        self.linear2 = torch.nn.Linear(4,2)
        self.linear3 = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x = x.view(-1, 6)
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    
model = Net()

criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

epoch_list = []
loss_list = []

def train(epoch):
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        running_loss = loss.item()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(running_loss)
    
    epoch_list.append(epoch)    
    loss_list.append(running_loss)    
    
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, target in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    # print('Accuracy on test set: %d  %%' % (100*correct/total))

for epoch in range(1):
    train(epoch)
    # test()

# plt.plot(epoch_list, loss_list)
# plt.show()