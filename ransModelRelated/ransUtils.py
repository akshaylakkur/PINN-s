import os
import ast
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ransFunction import *


# Load data and pre-processing
def load_single_file(file_path):
    ransData = pd.read_csv(file_path, quotechar='"')
    ransData.columns = ransData.columns.str.strip()

    numerical_features = ['x', 'y', 'p', 'Ux', 'Uy', 'Rx', 'Ry', 'Rxy']
    coeffs_columns = ['coeffsU1', 'coeffsL1', 'coeffsU2', 'coeffsL2']
    viscosity_column = 'viscosity(nut)'

    for col in coeffs_columns:
        ransData[col] = ransData[col].apply(ast.literal_eval)

    input_features = ['x', 'y']
    output_features = ['p', 'Ux', 'Uy', 'Rx', 'Ry', 'Rxy']

    scaler_input = StandardScaler()
    scaler_output = StandardScaler()

    ransData[input_features] = scaler_input.fit_transform(ransData[input_features])
    ransData[output_features] = scaler_output.fit_transform(ransData[output_features])

    tensors = {col: torch.tensor(ransData[col].values, dtype=torch.float32).unsqueeze(1).requires_grad_(True) for col in numerical_features}
    coeffs_tensors = {col: torch.tensor(ransData[col].tolist(), dtype=torch.float32).requires_grad_(True) for col in coeffs_columns}
    viscosity = torch.tensor(ransData[viscosity_column].values, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

    inputs = torch.cat(
        [tensors['x'], tensors['y']] + [coeffs_tensors[col] for col in coeffs_columns],
        dim=1
    )

    targets = torch.cat(
        [tensors[col] for col in output_features],
        dim=1
    )

    return {**tensors, **coeffs_tensors, "viscosity": viscosity, "inputs": inputs, "targets": targets, "scaler_input": scaler_input, "scaler_output": scaler_output}

# Custom loss
def custom_loss(outputs, targets, x, y, pred_Ux, pred_Uy, viscosity, pred_Rx, pred_Ry, pred_Rxy):
    mse_loss = nn.MSELoss()(outputs, targets)
    c = continuity(x, y, pred_Ux, pred_Uy)
    u_momentum = momentum_u(x, y, pred_Ux, pred_Uy, 1, outputs[:, 0:1], viscosity, pred_Rx, pred_Rxy)
    v_momentum = momentum_v(x, y, pred_Ux, pred_Uy, 1, outputs[:, 0:1], viscosity, pred_Ry, pred_Rxy)
    pinn_loss = c + u_momentum + v_momentum
    return mse_loss + 0.1 * pinn_loss

# Get files from training data path and sub-folders, with file name of certain pattern, e.g., *_iteration500.csv
def get_csv_files(training_data_path, file_name_pattern):
    return [os.path.join(root, file) for root, _, files in os.walk(training_data_path) for file in files if file.endswith(file_name_pattern)]


xlim = (-0.8, 1.3)
ylim = (-0.4, 0.4)

# Plot pressure
def plot_pressure(x, y, p, epoch, title, scaler_output, pngFilePrefix):
    x_np = x.detach().numpy()
    y_np = y.detach().numpy()
    points = np.column_stack((x_np, y_np))

    # Reshape p to match the scaler's expected input shape
    p_reshaped = p.detach().numpy().reshape(-1, 1)

    # Create a dummy array with 6 columns to match the scaler's output shape
    dummy_array = np.zeros((p_reshaped.shape[0], 6))
    dummy_array[:, 0] = p_reshaped.flatten()

    # Inverse transform the dummy array
    p_inverse = scaler_output.inverse_transform(dummy_array)[:, 0]

    grid_x, grid_y = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
    grid_p = griddata(points, p_inverse, (grid_x, grid_y), method='linear')

    plt.figure(figsize=(10, 8))
    plt.contourf(grid_x, grid_y, grid_p, levels=100, cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.title(title)
    plt.savefig(f'{pngFilePrefix}_epoch_{epoch}.png')
    plt.close()