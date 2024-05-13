import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from funcshape.networks import CurveReparametrizer
from funcshape.layers.sineseries import SineSeries
from funcshape.logging import Logger
from funcshape.reparametrize import reparametrize
from funcshape.utils import col_linspace
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from funcshape.loss import ShapeDistanceBase 

from typing import Tuple

class Function:
    def __init__(self, x, fx):
        self.x = x
        self.fx = fx
        coeffs = natural_cubic_spline_coeffs(self.x, self.fx)
        self.spline = NaturalCubicSpline(coeffs)

    def __call__(self, x):
        return self.spline.evaluate(x)

    def derivative(self, x, h=None):
        return self.spline.derivative(x, order=1)
    
    def compose(self, f):
        fx = f(self.x)
        y = self.spline.evaluate(fx).squeeze(-1)
        
        return Function(self.x, y)
    
def plot_function(f, npoints=201, dotpoints=None, ax=None, **kwargs):
    x = torch.linspace(0, 1, npoints).squeeze()
    fx = f(x).squeeze()

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, fx, **kwargs)

    if dotpoints is not None:
        x = torch.linspace(0, 1, dotpoints).squeeze()
        fx = f(x).squeeze()
        ax.plot(
            x,
            fx,
            c=ax.lines[-1].get_color(),
            ls="none",
            marker="o",
            markeredgecolor="black",
        )

    return ax

class SRSF:
    """SRSF transformation of functions"""

    def __init__(self, f : Function)-> None:
        self.f = f

    def __call__(self, x, h=None):
        grad = self.f.derivative(x)
        q = grad / torch.sqrt(torch.abs(grad) + 1e-3)
        
        return q
    

class FunctionDistance(ShapeDistanceBase):
    def create_point_collection(self, k):
        return col_linspace(0, 1, k)

    def get_determinant(self, network):
        return network.derivative(self.X, self.h)

    def loss_func(self, U, Y):
        error = ((self.Q.squeeze(-1) - torch.sqrt(U + 1e-8) * self.r(Y, self.h).squeeze(-1)) ** 2)
        l2_norm = torch.trapezoid(error.squeeze(), x=self.X.squeeze())
        
        return l2_norm
    
def get_warping_function(f1 : Function, f2 : Function, **kwargs)->Tuple[Function, CurveReparametrizer , np.ndarray]:
    q1, q2 = SRSF(f1), SRSF(f2)
    # Define loss, optimizer and run reparametrization.
    n_domain = kwargs.get("n_domain", 1024)
    loss_func = FunctionDistance(q1, q2, k=n_domain)

    best_error_value = np.inf
    for _ in range(kwargs.get("n_restarts", 10)):
        # Create reparametrization network
        RN = CurveReparametrizer([
            SineSeries(kwargs.get("n_basis", 20)) for i in range(kwargs.get("n_basis", 10))
        ])

        optimizer = optim.Adam(RN.parameters(), lr=kwargs.get("lr", 3e-4))
        error = reparametrize(RN, loss_func, optimizer, kwargs.get("n_iters", 100), Logger(0))

        if error[-1]<best_error_value:
            best_error = error
            best_error_value = best_error[-1]
            best_RN = RN
            print("Current best error : %2.4f"%best_error_value)

        if best_error_value<kwargs.get("eps", 1e-3):
            break

    # Get plot data to visualize diffeomorphism
    best_RN.detach()
    x = col_linspace(0, 1, n_domain)
    y = best_RN(x)
    print(x.shape, y.shape)

    return Function(x.squeeze(), y), best_RN, best_error