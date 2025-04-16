import torch


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
    return torch.mean(torch.abs(momentum_v_residual))






