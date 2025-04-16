import torch
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from aso_pinn.ransModelRelated.ransUtilsV2 import ransUtils

# Initialize RANS utils and load data
ransutils = ransUtils()
file_nums = [6412, 6612, 6912, 7412, 7612, 7912, 8612, 8908, 9405, 9612]
for file_num in file_nums:
    coords = np.loadtxt(f'/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/Training_Data/Naca{file_num}/Naca{file_num}Coords.csv')
    data = pd.read_csv(f'/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/Training_Data/Naca{file_num}/Naca{file_num}_iteration500.csv', quotechar='"')
    data.columns = data.columns.str.strip()

    # Split the input data
    x, y, p, Ux, Uy, Rx, Ry, Rxy, viscosity, coeffs = ransutils.split_input(data)
    x_np = x.detach().numpy()
    y_np = y.detach().numpy()
    points = np.column_stack((x_np, y_np))

    # Convert tensors to numpy arrays
    p_np = p.detach().numpy().flatten()
    Ux_np = Ux.detach().numpy().flatten()
    Uy_np = Uy.detach().numpy().flatten()
    Rx_np = Rx.detach().numpy().flatten()
    Ry_np = Ry.detach().numpy().flatten()
    Rxy_np = Rxy.detach().numpy().flatten()
    viscosity_np = viscosity.detach().numpy().flatten()


    def interpolate(points, values, target_x, target_y):
        target_point = np.array([[target_x, target_y]])
        # Perform interpolation using griddata
        # method='linear' for linear interpolation
        interpolated_value = griddata(points, values, target_point, method='linear')
        return interpolated_value[0]


    # Create numpy arrays for storing interpolated values
    n_points = len(coords)
    interpolated_p = np.zeros(n_points)
    interpolated_Ux = np.zeros(n_points)
    interpolated_Uy = np.zeros(n_points)
    interpolated_Rx = np.zeros(n_points)
    interpolated_Ry = np.zeros(n_points)
    interpolated_Rxy = np.zeros(n_points)
    interpolated_viscosity = np.zeros(n_points)

    # Perform interpolation for each variable
    for idx, coord in enumerate(coords):
        interpolated_p[idx] = interpolate(points, p_np, coord[0], coord[1])
        interpolated_Ux[idx] = interpolate(points, Ux_np, coord[0], coord[1])
        interpolated_Uy[idx] = interpolate(points, Uy_np, coord[0], coord[1])
        interpolated_Rx[idx] = interpolate(points, Rx_np, coord[0], coord[1])
        interpolated_Ry[idx] = interpolate(points, Ry_np, coord[0], coord[1])
        interpolated_Rxy[idx] = interpolate(points, Rxy_np, coord[0], coord[1])
        interpolated_viscosity[idx] = interpolate(points, viscosity_np, coord[0], coord[1])
        print(f'airfoil num: {file_num}', interpolated_p[idx], interpolated_Ux[idx], interpolated_Uy[idx], interpolated_Rx[idx], interpolated_Ry[idx], interpolated_Rxy[idx], interpolated_viscosity[idx])

    # Convert back to PyTorch tensors
    interpolated_p_tensor = torch.tensor(interpolated_p, dtype=torch.float32)
    interpolated_Ux_tensor = torch.tensor(interpolated_Ux, dtype=torch.float32)
    interpolated_Uy_tensor = torch.tensor(interpolated_Uy, dtype=torch.float32)
    interpolated_Rx_tensor = torch.tensor(interpolated_Rx, dtype=torch.float32)
    interpolated_Ry_tensor = torch.tensor(interpolated_Ry, dtype=torch.float32)
    interpolated_Rxy_tensor = torch.tensor(interpolated_Rxy, dtype=torch.float32)
    interpolated_viscosity_tensor = torch.tensor(interpolated_viscosity, dtype=torch.float32)

    df = pd.DataFrame({
        "x": [coord[0] for coord in coords],
        "y": [coord[1] for coord in coords],
        "p": interpolated_p,
        "Ux": interpolated_Ux,
        "Uy": interpolated_Uy,
        "Rx": interpolated_Rx,
        "Ry": interpolated_Ry,
        "Rxy": interpolated_Rxy,
        "viscosity": interpolated_viscosity
    })

    # Save to CSV file
    csv_filename = f"/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/Training_Data/Naca{file_num}/interpolated_data{file_num}.csv"
    df.to_csv(csv_filename, index=False)
