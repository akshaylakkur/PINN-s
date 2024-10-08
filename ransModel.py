"Rans Model with Pinn terms"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader
from scipy.interpolate import griddata

# Data setup
def runModel(airfoilFile):
    ransData = pd.read_csv(f'{airfoilFile}')
    ransData.columns = ransData.columns.str.strip()
    ransData = pd.DataFrame(ransData)

    # Convert to lists for each specific column
    x = ransData['x'].to_list()
    y = ransData['y'].to_list()
    p = ransData['p'].to_list()
    Ux = ransData['Ux'].to_list()
    Uy = ransData['Uy'].to_list()
    Rx = ransData['Rx'].to_list()
    viscosity = ransData['viscosity(nut)'].to_list()
    Ry = ransData['Ry'].to_list()
    Rxy = ransData['Rxy'].to_list()

    # Convert to tensors
    x = torch.tensor(x).unsqueeze(1).float().requires_grad_(True)
    y = torch.tensor(y).unsqueeze(1).float().requires_grad_(True)
    p = torch.tensor(p).unsqueeze(1).float().requires_grad_(True)
    Ux = torch.tensor(Ux).unsqueeze(1).float().requires_grad_(True)
    Uy = torch.tensor(Uy).unsqueeze(1).float().requires_grad_(True)
    Rx = torch.tensor(Rx).unsqueeze(1).float().requires_grad_(True)
    Ry = torch.tensor(Ry).unsqueeze(1).float().requires_grad_(True)
    Rxy = torch.tensor(Rxy).unsqueeze(1).float().requires_grad_(True)
    viscosity = torch.tensor(viscosity).unsqueeze(1).float().requires_grad_(True)


    # NN architecture
    class ransModel(nn.Module):
        def __init__(self):
            super(ransModel, self).__init__()
            self.layer1 = nn.Linear(2, 100)
            self.layer2 = nn.Linear(100, 100)
            self.layer5 = nn.Linear(100, 100)
            self.layer6 = nn.Linear(100, 6)

        def forward(self, x, y):
            inputs = torch.cat((x, y), dim=1)
            x = torch.relu(self.layer1(inputs))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer5(x))
            x = self.layer6(x)
            return x


    # Initialize the model and optimizer
    ransModel = ransModel()
    optimizer = optim.AdamW(ransModel.parameters(), lr=0.01)
    basicLoss = nn.MSELoss()
    train_loss = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ransModel = ransModel.to('cpu')

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

    # Plot actual pressure
    plt.figure()
    plt.contourf(grid_x, grid_y, grid_p, levels=100, cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.title(f'Pressure Distribution at Epoch')
    plt.show()

    # Define RANS equations as custom loss functions
    def continuity(x, y, Ux, Uy):
        dUx_dx = torch.autograd.grad(Ux, x, torch.ones_like(Ux), create_graph=True)[0]
        dUy_dy = torch.autograd.grad(Uy, y, torch.ones_like(Uy), create_graph=True)[0]
        continuity_residual = dUx_dx + dUy_dy
        return torch.mean(torch.abs(continuity_residual))

    def momentum_u(x, y, Ux, Uy, rho, p, nu, Rx, Rxy):
        dUx_dx = torch.autograd.grad(Ux, x, torch.ones_like(Ux), create_graph=True)[0]
        dUx_dy = torch.autograd.grad(Ux, y, torch.ones_like(Ux), create_graph=True)[0]
        dUy_dx = torch.autograd.grad(Uy, x, torch.ones_like(Uy), create_graph=True)[0]
        dP_dx = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]

        d2Ux_dx2 = torch.autograd.grad(dUx_dx, x, torch.ones_like(dUx_dx), create_graph=True)[0]
        d2Ux_dy2 = torch.autograd.grad(dUx_dy, y, torch.ones_like(dUx_dy), create_graph=True)[0]

        # du2_dx = torch.autograd.grad(Rx, x, torch.ones_like(Rx), create_graph=True)[0]
        # duv_dy = torch.autograd.grad(Rxy, y, torch.ones_like(Rxy), create_graph=True)[0]

        momentum_u_residual = Ux * dUx_dx + Uy * dUx_dy + (1 / rho) * dP_dx - nu * (
                    d2Ux_dx2 + d2Ux_dy2) + Rx
        return torch.mean(torch.abs(momentum_u_residual))

    def momentum_v(x, y, Ux, Uy, rho, p, nu, Ry, Rxy):
        dVy_dx = torch.autograd.grad(Uy, x, torch.ones_like(Ux), create_graph=True)[0]
        dVy_dy = torch.autograd.grad(Uy, y, torch.ones_like(Uy), create_graph=True)[0]
        dP_dy = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]

        d2Vy_dx2 = torch.autograd.grad(dVy_dx, x, torch.ones_like(dVy_dx), create_graph=True)[0]
        d2Vy_dy2 = torch.autograd.grad(dVy_dy, y, torch.ones_like(dVy_dy), create_graph=True)[0]

        # duv_dx = torch.autograd.grad(Rxy, x, torch.ones_like(Rxy), create_graph=True)[0]
        # dv2_dy = torch.autograd.grad(Ry, y, torch.ones_like(Ry), create_graph=True)[0]

        momentum_v_residual = Ux * dVy_dx + Uy * dVy_dy + (1 / rho) * dP_dy - nu * (
                    d2Vy_dx2 + d2Vy_dy2) + Ry
        return torch.mean(torch.abs(momentum_v_residual))  # Add torch.abs maybe


    # Training Loop
    def ransTrain(epochs):
        ransModel.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = ransModel(x, y)

            # Splitting the outputs to match the physical variables
            pred_p, pred_Ux, pred_Uy, pred_Rx, pred_Ry, pred_Rxy = outputs[:, 0:1], outputs[:, 1:2], outputs[:,2:3], outputs[:,3:4], outputs[:,4:5], outputs[:,5:6]

            # MSE Loss between predicted and true values
            traditionalLoss = basicLoss(outputs, torch.cat((p, Ux, Uy, Rx, Ry, Rxy), dim=1))

            # Compute residuals
            c = continuity(x, y, pred_Ux, pred_Uy)
            u_momentum = momentum_u(x, y, pred_Ux, pred_Uy, 1, pred_p, viscosity, pred_Rx, pred_Rxy)
            v_momentum = momentum_v(x, y, pred_Ux, pred_Uy, 1, pred_p, viscosity, pred_Ry, pred_Rxy)

            # Total PINN loss
            pinnLoss = c + u_momentum + v_momentum

            # Combine losses
            ransLoss = traditionalLoss + 0.01 * pinnLoss

            # Backpropagation and optimization
            ransLoss.backward()
            optimizer.step()

            if epoch % 250 == 0:
                print(f'Epoch {epoch} - Total Loss: {ransLoss.item()}, Continuity Loss: {c.item()}, U Momentum Loss: {u_momentum.item()}, V Momentum Loss: {v_momentum.item()}, MSE Loss: {traditionalLoss.item()}')

                # Convert pressure values and ensure they are 1D
                pred_p_np = pred_p.detach().numpy().flatten()

                # Interpolate pressure values onto the grid
                grid_p_pred = griddata(points, pred_p_np, (grid_x, grid_y), method='linear')

                # Plot predicted pressure
                plt.figure()
                plt.contourf(grid_x, grid_y, grid_p_pred, levels=100, cmap='viridis')
                plt.colorbar(label='Pressure Predicted')
                plt.xlim(xlim[0], xlim[1])
                plt.ylim(ylim[0], ylim[1])
                plt.title(f'Pressure Prediction Distribution at Epoch {epoch}')
                plt.show()



    # Start training
    ransTrain(2000)

    # Save the model state dictionary
    PATH = 'ransModel.pth'
    torch.save({
        'model_state_dict': ransModel.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)

    print("Model saved successfully.")




runModel(input('Input file name: '))



