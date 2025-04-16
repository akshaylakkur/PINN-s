import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from ransUtilsV2 import *
import matplotlib.pyplot as plt
from torch.nn import functional as F

ransUtils = ransUtils()
data = ransUtils.get_file_setup()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ransModel(nn.Module):
    def __init__(self, coeffs, coords, hidden_dim, out_dim):
        super(ransModel, self).__init__()
        self.register_buffer('fixed_coeffs', coeffs)
        self.fc_conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=coords,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                padding_mode='zeros'
            ),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                padding_mode='zeros'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc_conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=2,
                stride=1,
                padding=1,
                bias=False,
                padding_mode='zeros'
            ),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=2,
                stride=2,
                padding=1,
                bias=False,
                padding_mode='zeros'
            ),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc_conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                padding_mode='zeros'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.preds = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim*177, hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.Linear(hidden_dim//4, length * out_dim)
        )
    def change_buffer(self, new_val):
        self.fixed_coeffs =  new_val
        return self.fixed_coeffs

    def forward(self, t):
        t = self.fc_conv1(t)
        t = self.fc_conv2(t)
        t = self.fc_conv3(t)
        self.fixed_coeffs = F.pad(self.fixed_coeffs, (0, t.shape[2] - self.fixed_coeffs.shape[2], 0, t.shape[1] - self.fixed_coeffs.shape[1]))
        t = t + self.fixed_coeffs
        t = self.preds(t).view(1,length,7)
        return t



length = 11361
dummy_coeffs = torch.randn(1,1,32)
model = ransModel(dummy_coeffs, 2,30,7).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
epochs = 2011
loss_val_basic, loss_momentum_u, loss_momentum_v, loss_continuity, loss_total = [], [], [], [], []
airfoil_tracking = [2412, 2912, 4412, 4612, 4912, 5612, 5912, 6412, 6612, 6912, 7412, 7612, 8612, 8908, 9405, 9612]
def ransTrain():
    track_num = 0
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        model.train()
        for item in data:
            optimizer.zero_grad()
            x, y, p, ux, uy, rx, ry, rxy, viscosity, coefficients = ransUtils.split_input(item)
            x,y,p,ux,uy,rx,ry,rxy,viscosity,coefficients = x.to(device), y.to(device), p.to(device), ux.to(device), uy.to(device), rx.to(device), ry.to(device), rxy.to(device), viscosity.to(device), coefficients.to(device)
            inp = torch.cat([x,y], dim=1).unsqueeze(0).permute(0,2,1).to(device)
            model.change_buffer(coefficients)
            p_pred, ux_pred, uy_pred, rx_pred, ry_pred, rxy_pred, viscosity_pred = torch.split(model(inp), 1, dim=-1)
            loss_targets = torch.cat([p, ux, uy, rx, ry, rxy, viscosity], dim=1).to(device)
            loss_preds = torch.cat([p_pred, ux_pred, uy_pred, rx_pred, ry_pred, rxy_pred, viscosity_pred], dim=1).view(-1).view(length,7).to(device)
            orig_loss = loss_fn(loss_preds, loss_targets)
            continuity = ransUtils.continuity(x, y, ux_pred, uy_pred)
            momentum_u = ransUtils.momentum_u(x, y, ux_pred, uy_pred, 1, p_pred, viscosity, rx_pred, rxy_pred)
            momentum_v = ransUtils.momentum_v(x, y, uy_pred, ux_pred, 1, p_pred, viscosity, ry_pred, rxy_pred)
            total_loss = orig_loss + (continuity + momentum_u + momentum_v)
            total_loss.backward()
            optimizer.step()
            loss_val_basic.append(orig_loss.item())
            loss_momentum_u.append(momentum_u.item())
            loss_momentum_v.append(momentum_v.item())
            loss_continuity.append(continuity.item())
            loss_total.append(total_loss.item())
            if epoch % 10 == 0:
                print(f'airfoil: {airfoil_tracking[track_num]}, basic loss: {orig_loss.item()}, continuity_loss: {continuity.item()}, momentum_u_loss: {momentum_u.item()}, momentum_v_loss: {momentum_v.item()}, total_loss: {total_loss.item()}')
            track_num += 1
            torch.cuda.empty_cache()
            gc.collect()
        track_num = 0
    torch.save(model.state_dict(), 'ransModelV2.pth')

    return loss_val_basic, loss_momentum_u, loss_momentum_v, loss_continuity, loss_total

lvb, lmu, lmv, lc, lt = ransTrain()
    
plt.figure()
plt.plot(lvb)
plt.title('Basic Loss')
plt.grid(True)
plt.savefig('basic_loss.png')
plt.clf()

plt.figure()
plt.plot(lmu)
plt.title('Momentum U Loss')
plt.grid(True)
plt.savefig('momentum_u_loss.png')
plt.clf()

plt.figure()
plt.plot(lmv)
plt.title('Momentum V Loss')
plt.grid(True)
plt.savefig('momentum_v_loss.png')
plt.clf()

plt.figure()
plt.plot(lc)
plt.title('Continuity Loss')
plt.grid(True)
plt.savefig('continuity_loss.png')
plt.clf()

plt.figure()
plt.plot(lt)
plt.title('Total Loss')
plt.grid(True)
plt.savefig('total_loss.png')
plt.clf()
