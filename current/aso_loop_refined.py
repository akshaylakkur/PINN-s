import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np
from scipy.optimize import check_grad, approx_fprime, minimize
import scipy


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

sample_airfoil_file = pd.read_csv(sample_airfoil_file, quotechar='"')
x, y_, p, Ux, Uy, Rx, Ry, Rxy, viscosity = split_input(sample_airfoil_file)
count = 0

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

df = pd.read_csv(r"/Users/akshaylakkur/PycharmProjects/ASO/aso_pinn/Training_Data/Naca8908/Naca8908_iteration500.csv")
class AirfoilPlotter:
    def __init__(self, y):
        # yp = y.squeeze().detach().cpu().numpy().copy()
        plot_airfoil_points(y)
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
def to_numpy(tensor):
    return tensor.detach().cpu().numpy().copy()  # Explicit copy avoids warnings
def compute_lift_drag_direct(x, y, p, Ux, Uy, Rxy, viscosity, rho=1.225, U_inf=1.0, chord_length=1.0):
    x_np = to_numpy(x).flatten()
    y_np = to_numpy(torch.FloatTensor(y_).unsqueeze(1)).flatten()
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
def compute_lift_and_drag(y):
    with torch.no_grad():
        model.eval()
        outputs = model(x, torch.FloatTensor(y).unsqueeze(1))
        pred_p, pred_Ux, pred_Uy, pred_Rx, pred_Ry, pred_Rxy, pred_viscosity = outputs[:, 0:1], outputs[:,
                                                                                                1:2], outputs[:,
                                                                                                      2:3], outputs[
                                                                                                            :,
                                                                                                            3:4], outputs[
                                                                                                                  :,
                                                                                                                  4:5], outputs[
                                                                                                                        :,
                                                                                                                        5:6], outputs[
                                                                                                                              :,
                                                                                                                              6:7]
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("Bad input: NaN or Inf in y_surface")
        return 1e6
    else:
        L, D = compute_lift_drag_direct(x, y, pred_p, pred_Ux, pred_Uy, pred_Rxy, pred_viscosity)
        if np.abs(L) < 1e-8:
            print(f"Lift too small: L={L}, D={D}")
            return 1e6  # Or some penalty
        AirfoilPlotter(y)
        dl = np.abs(D / L)
        print(dl)
        return dl
def finite_diff_grad(y):
    """Wrapper for numerical gradient"""
    return approx_fprime(y, compute_lift_and_drag, epsilon=1e-8)  # Adjust epsilon if needed
plot_airfoil_points(y_points=y_.squeeze().detach().cpu().numpy().copy())
print(compute_lift_and_drag(y_.squeeze().detach().cpu().numpy().copy()))
y0 = y_.squeeze().detach().cpu().numpy().copy()
grad_error = check_grad(compute_lift_and_drag, finite_diff_grad, y0)

if grad_error < 1e-4:
    res = minimize(compute_lift_and_drag, y0, method='BFGS', options={'maxiter': 5})
else:
    print("WARNING: Numerical gradients unreliable! Fix before optimizing.")
np.seterr(all='raise')
try:
    res = minimize(compute_lift_and_drag, y_.squeeze().detach().cpu().numpy().copy(),
                                  method='Nelder-Mead', options={'maxiter': 5})
    print(res)
except FloatingPointError as e:
    print(f"Floating point error: {e}")