import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import ast

class ransUtils:
    def __init__(self):
        super(ransUtils, self).__init__()
        pass

    def continuity(self, x, y, Ux, Uy):
        dUx_dx = torch.autograd.grad(Ux, x, torch.ones_like(Ux), create_graph=True)[0]
        dUy_dy = torch.autograd.grad(Uy, y, torch.ones_like(Uy), create_graph=True)[0]
        continuity_residual = dUx_dx + dUy_dy
        return torch.mean(torch.abs(continuity_residual))

    def momentum_u(self, x, y, Ux, Uy, rho, p, nu, Rxx, Rxy):
        # Compute gradients of velocity components
        dUx_dx = torch.autograd.grad(Ux, x, torch.ones_like(Ux), create_graph=True)[0]
        dUx_dy = torch.autograd.grad(Ux, y, torch.ones_like(Ux), create_graph=True)[0]

        # Compute pressure gradient
        dP_dx = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]

        # Compute second derivatives of velocity components
        d2Ux_dx2 = torch.autograd.grad(dUx_dx, x, torch.ones_like(dUx_dx), create_graph=True)[0]
        d2Ux_dy2 = torch.autograd.grad(dUx_dy, y, torch.ones_like(dUx_dy), create_graph=True)[0]

        # Compute gradients of Reynolds stress components
        dRxx_dx = torch.autograd.grad(Rxx, x, torch.ones_like(Rxx), create_graph=True)[0]
        dRxy_dy = torch.autograd.grad(Rxy, y, torch.ones_like(Rxy), create_graph=True)[0]

        # Compute the residual for the x-momentum equation
        momentum_u_residual = (
                Ux * dUx_dx + Uy * dUx_dy  # Convective terms
                + (1 / rho) * dP_dx  # Pressure gradient
                - nu * (d2Ux_dx2 + d2Ux_dy2)  # Viscous terms
                - dRxx_dx - dRxy_dy  # Reynolds stress terms
        )
        return torch.mean(torch.abs(momentum_u_residual))

    def momentum_v(self, x, y, Ux, Uy, rho, p, nu, Rxy, Ryy):
        # Compute gradients of velocity components
        dVy_dx = torch.autograd.grad(Uy, x, torch.ones_like(Uy), create_graph=True)[0]
        dVy_dy = torch.autograd.grad(Uy, y, torch.ones_like(Uy), create_graph=True)[0]

        # Compute pressure gradient
        dP_dy = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]

        # Compute second derivatives of velocity components
        d2Vy_dx2 = torch.autograd.grad(dVy_dx, x, torch.ones_like(dVy_dx), create_graph=True)[0]
        d2Vy_dy2 = torch.autograd.grad(dVy_dy, y, torch.ones_like(dVy_dy), create_graph=True)[0]

        # Compute gradients of Reynolds stress components
        dRxy_dx = torch.autograd.grad(Rxy, x, torch.ones_like(Rxy), create_graph=True)[0]
        dRyy_dy = torch.autograd.grad(Ryy, y, torch.ones_like(Ryy), create_graph=True)[0]

        # Compute the residual for the y-momentum equation
        momentum_v_residual = (
                Ux * dVy_dx + Uy * dVy_dy  # Convective terms
                + (1 / rho) * dP_dy  # Pressure gradient
                - nu * (d2Vy_dx2 + d2Vy_dy2)  # Viscous terms
                - dRxy_dx - dRyy_dy  # Reynolds stress terms
        )
        return torch.mean(torch.abs(momentum_v_residual))

    def custom_loss(self, outputs, targets, x, y, pred_Ux, pred_Uy, viscosity, pred_Rx, pred_Ry, pred_Rxy):
        mse_loss = nn.MSELoss()(outputs, targets)
        c = continuity(x, y, pred_Ux, pred_Uy)
        u_momentum = momentum_u(x, y, pred_Ux, pred_Uy, 1, outputs[:, 0:1], viscosity, pred_Rx, pred_Rxy)
        v_momentum = momentum_v(x, y, pred_Ux, pred_Uy, 1, outputs[:, 0:1], viscosity, pred_Ry, pred_Rxy)
        pinn_loss = c + u_momentum + v_momentum
        return mse_loss + 0.1 * pinn_loss

    def split_input(self, ransData):
        x = torch.tensor(ransData['x'].to_list()).unsqueeze(1).float().requires_grad_(True)
        y = torch.tensor(ransData['y'].to_list()).unsqueeze(1).float().requires_grad_(True)
        p = torch.tensor(ransData['p'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Ux = torch.tensor(ransData['Ux'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Uy = torch.tensor(ransData['Uy'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Rx = torch.tensor(ransData['Rx'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Ry = torch.tensor(ransData['viscosity'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Rxy = torch.tensor(ransData['Ry'].to_list()).unsqueeze(1).float().requires_grad_(True)
        viscosity = torch.tensor(ransData['Rxy'].to_list()).unsqueeze(1).float().requires_grad_(True)
        # coeffsU1 = torch.tensor(ransData['coeffsU1'].apply(ast.literal_eval)).float().requires_grad_(True)[0]
        # coeffsL1 = torch.tensor(ransData['coeffsL1'].apply(ast.literal_eval)).float().requires_grad_(True)[0]
        # coeffsU2 = torch.tensor(ransData['coeffsU2'].apply(ast.literal_eval)).float().requires_grad_(True)[0]
        # coeffsL2 = torch.tensor(ransData['coeffsL2'].apply(ast.literal_eval)).float().requires_grad_(True)[0]
        # coeffs = torch.cat([coeffsU1, coeffsL1, coeffsU2, coeffsL2], dim=-1).unsqueeze(0).unsqueeze(0)
        return x, y, p, Ux, Uy, Rx, Ry, Rxy, viscosity#, coeffs

    def get_file_setup(self):
        file_nums = [2412, 2912, 4412, 4612, 4912, 5412, 5612, 5912, 6412, 6612, 6912, 7412, 7612, 7912, 8612, 8908,
                    9405, 9612]
        file_paths = []
        for file in file_nums:
            data = f'/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/Training_Data/Naca{file}/interpolated_data{file}.csv'
            #data = f'/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/Training_Data/Naca{file}/Naca{file}_iteration500.csv'
            data = pd.read_csv(data, quotechar='"')
            data.columns = data.columns.str.strip()
            data = pd.DataFrame(data)
            file_paths.append(data)
        return file_paths

'testing airfoils:'
'5412, 7912'