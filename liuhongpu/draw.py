import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y = [3.0,5.0,7.0,9.0,11.0,13.0]

def forward(x):
    return x*w+b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y) * (y_pred-y)

w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 1.1, 0.1):
    mse_sub = []
    for b in np.arange(0.0,1.1,0.1):
        # print("w=", w, "b=", b)
        l_sum = 0
        for x_val, y_val in zip(x,y):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            # print('\t', x_val, y_val, y_pred_val, loss_val)
        # print('MSE=', l_sum/6)
        w_list.append(w)
        b_list.append(b)
        mse_sub.append(l_sum/6)
    mse_list.append(mse_sub)

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
W,B = np.meshgrid(w_list,b_list)

ax.plot_wireframe(W,B,np.array(mse_list))
plt.show()

# x = [1,2,3]
# y = [4,5,6]
# z=[7,8,9]
# x, y = np.meshgrid(x,y)
# z = np.sqrt(x**2 + y**2)
# print(x, y, z)

# print(np.meshgrid(x))
