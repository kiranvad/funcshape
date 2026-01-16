import torch
from scipy.interpolate import interp1d
import numpy as np
from typing import Callable

from funcshape.derivatives import central_differences
from funcshape.diffeomorphism import Diffeomorphism1D


class Curve:
    """Define a torch-compatible parametrized curve class, with finite
    difference approximation of derivatives, and composition operator."""

    def __init__(self, component_function_tuple):
        self.C = tuple(component_function_tuple)
        self.dim = len(self.C)

    def __call__(self, X):
        return torch.cat([ci(X) for ci in self.C], dim=-1)

    def derivative(self, X, h):
        return torch.cat([central_differences(ci, X, h) for ci in self.C], dim=-1)

    def compose_component(self, i, f):
        return lambda x: self.C[i](f(x))

    def compose(self, f):
        return Curve((self.compose_component(i, f) for i in range(self.dim)))


class TorchCurve:
    def __init__(self, components: list[Callable]):
        self.components = components

    def __call__(self, x):
        return torch.stack([ci(x) for ci in self.components], dim=-1)

    def derivative(self, x, h):
        return torch.stack(
            [
                torch.autograd.grad(ci(x), x, torch.ones_like(x))[0]
                for ci in self.components
            ],
            dim=-1,
        )


class ComposedCurve(Curve):
    def __init__(self, curve: Curve, diffeomorphism: Diffeomorphism1D):
        self.c = curve
        self.diffeo = diffeomorphism

    def __call__(self, X):
        return self.c(self.diffeo(X))

    def derivative(self, X, h):
        return self.c.derivative(self.diffeo(X), h) * self.diffeo.derivative(X, h)

    @property
    def dim(self):
        return self.c.dim

class ParametricCurve(Curve):
    """
    Generic parametric curve.
    Takes evaluated points as input and returns a callable parametric curve.
    """
    def __init__(self, points: np.ndarray):
        """
        points: Tensor of shape (N, D) where D is the dimension (e.g., 2 for 2D, 3 for 3D)
        """
        assert points.ndim == 2, "points must be a 2D tensor of shape (N, D)"
        self.points = points
        self.N, self.D = points.shape

        # Parameterization: t in [0,1]
        self.t_values = torch.linspace(0, 1, self.N)

        # Create an interpolation function for each dimension
        self.interpolators = {}
        t_np = self.t_values.numpy()
        coords = ["x", "y"]
        for d in range(self.D):
            self.interpolators[coords[d]] = interp1d(
                t_np, 
                points[:, d], 
                kind='cubic', 
                fill_value="extrapolate"
            )
        
        def xfun(t):
            t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
            return (torch.tensor(self.interpolators["x"](t_np), dtype=torch.float32))

        def yfun(t):
            t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
            return (torch.tensor(self.interpolators["y"](t_np), dtype=torch.float32))

        super().__init__((xfun, yfun))

    def derivative(self, t, h=1e-5):
        """
        Numerical derivative using central differences
        """
        t = torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t
        t_plus = t + h
        t_minus = t - h
        f_plus = self(t_plus)
        f_minus = self(t_minus)
        return (f_plus - f_minus) / (2 * h)
