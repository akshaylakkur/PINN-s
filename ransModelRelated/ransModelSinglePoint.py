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

class ransModelSinglePoint(nn.Module):
    def __init__(self, coeffs, coords, hidden_dim, out_dim):
        super(ransModelSinglePoint, self).__init__()
        self.register_buffer('fixed_coeffs', coeffs)
        self.fc1 = nn.Sequential(
            nn.Linear(coords, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.preds = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, t):
        t = self.fc1(t)
        t = self.fc2(t)
        t = self.fc3(t)
        t = self.preds(t)
        return t

length = 11361
dummy_coeffs = torch.randn(1, 1, 32)
model = ransModelSinglePoint(dummy_coeffs, 2, 30, 7).to(device)
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
            x, y, p, ux, uy, rx, ry, rxy, viscosity, coefficients = x.to(device), y.to(device), p.to(device), ux.to(device), uy.to(device), rx.to(device), ry.to(device), rxy.to(device), viscosity.to(device), coefficients.to(device)
            inp = torch.cat([x, y], dim=1).to(device)
            predictions = []
            for i in range(inp.shape[0]):
                single_point = inp[i].unsqueeze(0)
                pred = model(single_point)
                predictions.append(pred)
            predictions = torch.stack(predictions, dim=0).squeeze(1)
            loss_targets = torch.cat([p, ux, uy, rx, ry, rxy, viscosity], dim=1).to(device)
            orig_loss = loss_fn(predictions, loss_targets)
            continuity = ransUtils.continuity(x, y, predictions[:, 1], predictions[:, 2])
            momentum_u = ransUtils.momentum_u(x, y, predictions[:, 1], predictions[:, 2], 1, predictions[:, 0], viscosity, predictions[:, 3], predictions[:, 5])
            momentum_v = ransUtils.momentum_v(x, y, predictions[:, 2], predictions[:, 1], 1, predictions[:, 0], viscosity, predictions[:, 4], predictions[:, 5])
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
    torch.save(model.state_dict(), 'ransModelSinglePoint.pth')
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