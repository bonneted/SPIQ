"""Backend supported: pytorch, jax, paddle

Implementation of the linear elasticity 2D example in paper https://doi.org/10.1016/j.cma.2021.113741.
References:
    https://github.com/sciann/sciann-applications/blob/master/SciANN-Elasticity/Elasticity-Forward.ipynb.
"""
import deepxde as dde
import numpy as np

lmbd = 1.0
mu = 0.5
Q = 4.0

# Define functions
sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack

geom = dde.geometry.Rectangle([0, 0], [1, 1])
BC_type = ["hard", "soft"][0]


def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)


def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.0)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)


# Exact solutions
def func(x):
    ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    uy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

    return np.hstack((ux, uy))

# Soft Boundary Conditions
def S_from_output(x, f,X):
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)[0]
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)[0]

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd

    return stack((S_xx, S_yy), axis=1)

ux_top_bc = dde.icbc.OperatorBC(geom, lambda x, f, X: f[0][:, 0], boundary_top)
ux_bottom_bc = dde.icbc.OperatorBC(geom, lambda x, f, X: f[0][:, 0], boundary_bottom)
uy_left_bc = dde.icbc.OperatorBC(geom, lambda x, f, X: f[0][:, 1], boundary_left)
uy_bottom_bc = dde.icbc.OperatorBC(geom, lambda x, f, X: f[0][:, 1], boundary_bottom)
uy_right_bc = dde.icbc.OperatorBC(geom, lambda x, f, X: f[0][:, 1], boundary_right)
sxx_left_bc = dde.icbc.OperatorBC(geom, lambda x, f, X: S_from_output(x, f, X)[:, 0], boundary_left)
sxx_right_bc = dde.icbc.OperatorBC(geom, lambda x, f, X: S_from_output(x, f, X)[:, 0], boundary_right)
syy_top_bc = dde.icbc.OperatorBC(geom, lambda x, f, X: S_from_output(x, f, X)[:, 1]-(lmbd + 2*mu)*Q*sin(np.pi*x[:,0:1]), boundary_top)

bcs = [ux_top_bc, ux_bottom_bc, uy_left_bc, uy_bottom_bc, uy_right_bc] if BC_type == "soft" else []
bcs += [
    sxx_left_bc,
    sxx_right_bc,
    syy_top_bc,
]

# Hard Boundary Conditions
def hard_BC(x, f):
    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]
    return stack((Ux, Uy), axis=1)


def fx(x):
    return (
        -lmbd
        * (
            4 * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - mu
        * (
            np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - 8 * mu * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
    )


def fy(x):
    return (
        lmbd
        * (
            3 * Q * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
            - 2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
        )
        - mu
        * (
            2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
            + (Q * x[:, 1:2] ** 4 * np.pi**2 * sin(np.pi * x[:, 0:1])) / 4
        )
        + 6 * Q * mu * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
    )
    

def pde(x, f):
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)
    E_xy = (0.5 * (dde.grad.jacobian(f, x, i=0, j=1)[0] + dde.grad.jacobian(f, x, i=1, j=0)[0]), lambda x: 0.5 * (dde.grad.jacobian(f, x.reshape(1,-1), i=0, j=1)[1](x) + dde.grad.jacobian(f, x.reshape(1,-1), i=1, j=0)[1](x))) 

    S_xx = (E_xx[0] * (2 * mu + lmbd) + E_yy[0] * lmbd, lambda x: E_xx[1](x) * (2 * mu + lmbd) + E_yy[1](x) * lmbd)
    S_yy = (E_yy[0] * (2 * mu + lmbd) + E_xx[0] * lmbd, lambda x: E_yy[1](x) * (2 * mu + lmbd) + E_xx[1](x) * lmbd)
    S_xy = (E_xy[0] * 2 * mu, lambda x: E_xy[1](x) * 2 * mu)

    Sxx_x = dde.grad.jacobian(S_xx, x, i=0, j=0)[0]
    Syy_y = dde.grad.jacobian(S_yy, x, i=0, j=1)[0]
    Sxy_x = dde.grad.jacobian(S_xy, x, i=0, j=0)[0]
    Sxy_y = dde.grad.jacobian(S_xy, x, i=0, j=1)[0]

    momentum_x = Sxx_x + Sxy_y - fx(x)
    momentum_y = Sxy_x + Syy_y - fy(x)

    return [momentum_x, momentum_y]

# def pde(x, f):
#     E_xx = dde.grad.jacobian(f, x, i=0, j=0)
#     E_yy = dde.grad.jacobian(f, x, i=1, j=1)
#     E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))

#     S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
#     S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
#     S_xy = E_xy * 2 * mu

#     Sxx_x = dde.grad.jacobian(S_xx, x, i=0, j=0)
#     Syy_y = dde.grad.jacobian(S_yy, x, i=0, j=1)
#     Sxy_x = dde.grad.jacobian(S_xy, x, i=0, j=0)
#     Sxy_y = dde.grad.jacobian(S_xy, x, i=0, j=1)

#     momentum_x = Sxx_x + Sxy_y - fx(x)
#     momentum_y = Sxy_x + Syy_y - fy(x)

#     return [momentum_x, momentum_y]


data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=500,
    num_boundary=500,
    solution=func,
    num_test=100,
)

layers = [2, [40] * 2, [40] * 2, [40] * 2, [40] * 2, 2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.PFNN(layers, activation, initializer)
if BC_type == "hard":
    net.apply_output_transform(hard_BC)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=5000, display_every=200)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
