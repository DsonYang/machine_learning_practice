import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import os
import matplotlib.pyplot as plt
from datetime import datetime


class DiabetesDataset(Dataset):
    def __init__(self):
        self.x_data = np.loadtxt('dataset/diabetes_data.csv.gz', delimiter=' ', dtype=np.float32)
        self.y_data = np.loadtxt('dataset/diabetes_target.csv.gz',ndmin=2 , dtype=np.float32)
        self.len = self.x_data.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(10, 9)
        self.linear11 = torch.nn.Linear(9, 8)
        self.linear2 = torch.nn.Linear(8, 7)
        self.linear22 = torch.nn.Linear(7, 6)
        self.linear3 = torch.nn.Linear(6, 5)
        self.linear33 = torch.nn.Linear(5, 4)
        self.linear4 = torch.nn.Linear(4, 3)
        self.linear44 = torch.nn.Linear(3, 2)
        self.linear5 = torch.nn.Linear(2,1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear11(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear22(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear33(x))
        x = self.activation(self.linear4(x))
        x = self.activation(self.linear44(x))
        x = self.activation(self.linear5(x))
        # x = self.sigmoid(x)
        return x
    
model = Model()

criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
epoch_list = []
loss_list = []
for epoch in range(1000):
    for i,data in enumerate(train_loader, 0):
        inputs, labels = data
        
        y_pred = model(inputs)
        # print(labels,y_pred)
        loss = criterion(y_pred,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_list.append(epoch)
    loss_list.append(loss.item()) 
    # print(epoch,loss.item())   

now = datetime.now()
os.environ['KMP_DUPLICATE_LIB_OK']='True' # to slove 'Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.'
plt.plot(epoch_list, loss_list)
plt.savefig("output_figures/liuhongpu/MultipleFeatureLinearRegre_"+now.strftime("%H_%M_%S")+"_shuffle_true.pdf")
print('done')
# plt.show()
