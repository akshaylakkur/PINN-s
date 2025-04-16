def split_input(ransData):
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
    return x, y, p, Ux, Uy, Rx, Ry, Rxy, viscosity  # , coeffs


import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


class Prediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Prediction, self).__init__()
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
        out = torch.cat((x, y), dim=-1)
        # out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


model = Prediction(2, 100, 7)
# model.load_state_dict(torch.load(r"C:\Users\dwbh2\OneDrive\Desktop\ASDRP\aso_pinn\current\PINNmodel.pth"))
# sample_airfoil_file = r"C:\Users\dwbh2\OneDrive\Desktop\ASDRP\aso_pinn\Training_Data\Naca8908\interpolated_data8908.csv"

model.load_state_dict(torch.load(r"/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/current/PINNmodel.pth"))
sample_airfoil_file = r"/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/Training_Data/Naca8908/interpolated_data8908.csv"


# model.load_state_dict(torch.load("current/PINNmodel.pth"))
# sample_airfoil_file = "Training_Data/Naca8908/interpolated_data8908.csv"


sample_airfoil_file = pd.read_csv(sample_airfoil_file, quotechar='"')
x, y_, p, Ux, Uy, Rx, Ry, Rxy, viscosity = split_input(sample_airfoil_file)


def plot_airfoil_points(y_points, iteration=None):
    """Plots airfoil shape with proper iteration tracking."""
    x_points = x.detach().cpu().numpy().copy()  # Fixed x-coordinates

    plt.figure(figsize=(12, 4))
    plt.plot(x_points, y_points, 'b-', linewidth=2, label='Airfoil')
    plt.plot([x_points.min(), x_points.max()], [0, 0], 'k--', alpha=0.3, label='Chord')

    # Highlight leading edge
    le_idx = np.argmin(x_points)
    plt.scatter(x_points[le_idx], y_points[le_idx], c='r', s=50, label='LE')

    plt.title(f"Iteration {iteration}" if iteration is not None else "Initial Airfoil")
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


class AirfoilPlotter:
    def __init__(self, max_plots=5):
        self.iteration = 0
        self.max_plots = max_plots
        self.last_y = None  # Track previous shape

    def __call__(self, y_points, *_):
        """Callback that ensures all iterations are captured."""
        # Only plot if shape changed significantly
        if self.last_y is None or np.max(np.abs(y_points - self.last_y)) > 1e-6:
            if self.iteration <= self.max_plots:
                plot_airfoil_points(y_points, self.iteration)
                self.last_y = y_points.copy()
            self.iteration += 1


def compute_lift_and_drag(y):
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("Bad input: NaN or Inf in y_surface")
        return 1e6
    # y = torch.tensor(y).unsqueeze(2)
    y = torch.FloatTensor(y).unsqueeze(1)

    with torch.no_grad():
        model.eval()
        outputs = model(x, y)
        pred_p, pred_Ux, pred_Uy, pred_Rx, pred_Ry, pred_Rxy, viscosity = outputs[:, 0:1], outputs[:, 1:2], outputs[:,
                                                                                                            2:3], outputs[
                                                                                                                  :,
                                                                                                                  3:4], outputs[
                                                                                                                        :,
                                                                                                                        4:5], outputs[
                                                                                                                              :,
                                                                                                                              5:6], outputs[
                                                                                                                                    :,
                                                                                                                                    6:7]
    # print (pred_p[:5])
    df = pd.read_csv(r"/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/Training_Data/Naca8908/Naca8908_iteration500.csv")

    # df = pd.read_csv("Training_Data/Naca8908/Naca8908_iteration500.csv")
    Rxy = torch.tensor(df[' Rxy'].values, dtype=torch.float32)

    def finite_difference2(values, points, i, direction):
        """Improved finite difference with smoother central differencing."""
        epsilon = 1e-8  # Prevent division by zero

        if direction == "x":
            if i == 0:
                dx = points[i + 1][0] - points[i][0]
                return (values[i + 1] - values[i]) / (dx + epsilon)
            elif i == len(values) - 1:
                dx = points[i][0] - points[i - 1][0]
                return (values[i] - values[i - 1]) / (dx + epsilon)
            else:
                dx1 = points[i][0] - points[i - 1][0]
                dx2 = points[i + 1][0] - points[i][0]
                return (dx2 * values[i - 1] - (dx1 + dx2) * values[i] + dx1 * values[i + 1]) / (dx1 * dx2 + epsilon)

        elif direction == "y":
            if i == 0:
                dy = points[i + 1][1] - points[i][1]
                return (values[i + 1] - values[i]) / (dy + epsilon)
            elif i == len(values) - 1:
                dy = points[i][1] - points[i - 1][1]
                return (values[i] - values[i - 1]) / (dy + epsilon)
            else:
                dy1 = points[i][1] - points[i - 1][1]
                dy2 = points[i + 1][1] - points[i][1]
                return (dy2 * values[i - 1] - (dy1 + dy2) * values[i] + dy1 * values[i + 1]) / (dy1 * dy2 + epsilon)

    # Usage example:
    # Assuming you have:
    # x, y: 2D grid coordinates (from meshgrid)
    # Ux, Uy: Velocity predictions on grid
    # airfoil_points: Nx2 array of your airfoil surface points

    # plot_velocity_field(x, y, Ux, Uy, p=p_pred, airfoil_points=airfoil_points)
    def compute_lift_drag_direct(x, y, p, Ux, Uy, Rxy, viscosity, rho=1.225, U_inf=1.0, chord_length=1.0):
        # Convert all inputs to detached numpy arrays first
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy().copy()  # Explicit copy avoids warnings

        x_np = to_numpy(x).flatten()
        y_np = to_numpy(y).flatten()
        p_np = to_numpy(p).flatten()
        Ux_np = to_numpy(Ux).flatten()
        Uy_np = to_numpy(Uy).flatten()
        Rxy_np = to_numpy(Rxy).flatten()
        if torch.is_tensor(viscosity):
            if viscosity.numel() == 1:
                visc_np = viscosity.item()
            else:
                visc_np = viscosity.detach().cpu().numpy().flatten()
        else:
            visc_np = viscosity

        points = np.column_stack((x_np, y_np))
        normals = []

        # 1. Compute surface normals (unchanged)
        for i in range(len(points)):
            # Segment connects points[i] to points[i+1] (with wrap-around)
            x_start, y_start = points[i]
            x_end, y_end = points[(i + 1) % len(points)]  # Wrap-around for closed shapes
            dx = x_end - x_start
            dy = y_end - y_start
            # Normal = outward-facing perpendicular vector
            normal_x = dy
            normal_y = -dx
            norm = np.sqrt(normal_x ** 2 + normal_y ** 2)
            if norm < 1e-8:
                print(f"Zero normal at point {i}: dx={dx}, dy={dy}")
                norm = 1e-8  # Avoid divide by zero
            normals.append([normal_x / norm, normal_y / norm])  # Unit normal
        normals = np.array(normals)

        # 2. Finite differences with explicit casting
        tau_xy = []

        for i in range(len(points)):
            dU_dy = finite_difference2(Ux_np, points, i, "y")
            dV_dx = finite_difference2(Uy_np, points, i, "x")
            # print(f"Point {i}: dx={dx:.4e}, dy={dy:.4e}, dU/dy={dU_dy:.4e}, dV/dx={dV_dx:.4e}")
            # Shear stress = Laminar + Turbulent part
            tau_xy_i = visc_np[i] * (dU_dy + dV_dx) - rho * Rxy_np[i]
            tau_xy.append(tau_xy_i)
        # 3. Force integration (unchanged)
        L, D = 0.0, 0.0
        for i in range(len(points) - 1):
            dS = np.sqrt((points[i + 1, 0] - points[i, 0]) ** 2 +
                         (points[i + 1, 1] - points[i, 1]) ** 2)
            nx, ny = normals[i]

            dF_px = -p_np[i] * nx * dS
            dF_py = -p_np[i] * ny * dS
            dF_tx = tau_xy[i] * ny * dS
            dF_ty = tau_xy[i] * nx * dS
            if np.isnan(dF_px) or np.isinf(dF_px):
                print(f"Bad force at segment {i}: p={p_np[i]}, nx={nx}, dS={dS}")
            D += dF_px + dF_tx
            L += dF_py + dF_ty
            ##print(f"Segment {i}: p={p_np[i]:.4f}, nx={normal_x:.4f}, ny={normal_y:.4f}, dF_px={dF_px:.4e}, dF_py={dF_py:.4e}")
        q_inf = 0.5 * rho * U_inf ** 2
        # print(f"q_inf: {0.5 * rho * U_inf**2}")
        # print(f"Min/Max p: {np.min(p_np)}, {np.max(p_np)}")
        # print(f"Min segment length: {min([np.linalg.norm(points[i+1]-points[i]) for i in range(len(points)-1)])}")
        return L / (q_inf * chord_length), D / (q_inf * chord_length)

    # plot_scattered_velocities(x.detach().cpu().numpy().copy(),y.detach().cpu().numpy().copy(), pred_Ux.detach().cpu().numpy().copy(), pred_Uy.detach().cpu().numpy().copy())
    L, D = compute_lift_drag_direct(x, y, pred_p, pred_Ux, pred_Uy, pred_Rxy, viscosity)
    if np.abs(L) < 1e-8:
        print(f"Lift too small: L={L}, D={D}")
        return 1e6  # Or some penalty
    dl = np.abs(D / L)
    print(dl)
    return dl


import scipy

plot_airfoil_points(y_points=y_.squeeze().detach().cpu().numpy().copy())
print(compute_lift_and_drag(y_.squeeze().detach().cpu().numpy().copy()))
import numpy as np
from scipy.optimize import check_grad, approx_fprime, minimize

# 1. Define your initial guess (y0) first
y0 = y_.squeeze().detach().cpu().numpy().copy()  # Your existing initialization


# 2. Add gradient verification here (BEFORE calling minimize)
def finite_diff_grad(y):
    """Wrapper for numerical gradient"""
    return approx_fprime(y, compute_lift_and_drag, epsilon=1e-8)  # Adjust epsilon if needed


grad_error = check_grad(compute_lift_and_drag, finite_diff_grad, y0)
print(f"Gradient consistency error: {grad_error:.2e}")  # Should be < 1e-5

# 3. Only proceed to optimization if gradient check passes
if grad_error < 1e-4:
    res = minimize(compute_lift_and_drag, y0, method='BFGS', options={'maxiter': 5})
else:
    print("WARNING: Numerical gradients unreliable! Fix before optimizing.")
np.seterr(all='raise')
plotter = AirfoilPlotter(max_plots=5)
try:
    res = minimize(compute_lift_and_drag, y_.squeeze().detach().cpu().numpy().copy(),
                                  method='Nelder-Mead', options={'maxiter': 5}, callback=plotter)
    print(res)
except FloatingPointError as e:
    print(f"Floating point error: {e}")
# print (scipy.optimize.minimize(compute_lift_and_drag, y_.squeeze().detach().cpu().numpy().copy(),  method='L-BFGS-B', options={'maxiter': 5}))
# print("")



