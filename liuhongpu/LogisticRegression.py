import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x_data = torch.Tensor([[5],[6],[7],[8],[9],[10]])
y_data = torch.Tensor([[0],[0],[0],[1],[1],[1]])

test_data = torch.Tensor([[4],[11]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    # print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(test_data))
# plt.plot(epoch_list, loss_list)
# plt.show()
