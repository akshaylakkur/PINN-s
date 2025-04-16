import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from aso_pinn.ransModelRelated.ransUtilsV2 import ransUtils
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class Prediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Prediction,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x, y):
        out = torch.cat((x, y), dim=1)
        #out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def plot(x, y, p, pred_p=None, epoch=None, actual=True):
    xlim = (-0.8, 1.3)
    ylim = (-0.4, 0.4)

    # Convert tensors to numpy arrays
    x_np = x.detach().numpy()
    y_np = y.detach().numpy()

    # Ensure x_np and y_np are 1D and stack them as points
    points = np.column_stack((x_np, y_np))

    # Convert pressure values and ensure they are 1D
    p_np = p.detach().numpy().flatten()


    # Define a grid for interpolation based on the given xlim and ylim
    grid_x, grid_y = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]

    # Interpolate pressure values onto the grid
    grid_p = griddata(points, p_np, (grid_x, grid_y), method='linear')


    if actual:
        # Plot predicted pressure
        pred_p_np = pred_p.detach().numpy().flatten()
        grid_p_pred = griddata(points, pred_p_np, (grid_x, grid_y), method='linear')
        plt.figure()
        plt.contourf(grid_x, grid_y, grid_p_pred, levels=100, cmap='viridis')
        plt.colorbar(label='Pressure Predicted')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title(f'Airfoil {file_nums[num]} Pressure Prediction Distribution at Epoch {epoch}')
        plt.show()
    else:
        plt.figure()
        plt.contourf(grid_x, grid_y, grid_p, levels=100, cmap='viridis')
        plt.colorbar(label='Pressure')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.title(f'True Pressure')
        plt.show()


model = Prediction(2, 100, 7)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 2001
model.train()
ransUtils = ransUtils()
data = ransUtils.get_file_setup()

file_nums = [2412, 2912, 4412, 4612, 4912, 5412, 5612, 5912, 6412, 6612, 6912, 7412, 7612, 7912, 8612, 8908, 9405, 9612]

for num in range(0,16):
    x, y, p, Ux, Uy, Rx, Ry, Rxy, viscosity = ransUtils.split_input(data[num])
    loss_list = []

    plot(x,y,p,actual=False)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x, y)
        pred_p, pred_Ux, pred_Uy, pred_Rx, pred_Ry, pred_Rxy, viscosity = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:,3:4], outputs[:,4:5], outputs[:,5:6], outputs[:,6:7]
        loss = loss_fn(outputs, torch.cat((p,Ux,Uy,Rx,Ry,Rxy,viscosity), dim=1))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if epoch % 250 == 0:
            plot(x,y,p,pred_p,epoch)
            print(loss.item())

    plt.figure()
    plt.plot(loss_list)
    plt.title(f'loss graph for airfoil {file_nums[num]}')
torch.save(model.state_dict(), 'PINNmodel.pth')