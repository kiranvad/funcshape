import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from funcshape.networks import CurveReparametrizer
from funcshape.layers.sineseries import SineSeries
from funcshape.logging import Logger
from funcshape.reparametrize import reparametrize
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from funcshape.loss import ShapeDistanceBase 
from funcshape.layers.layerbase import CurveLayer

from abc import abstractmethod 
from math import pi, sqrt
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
        self.function = f
        grad = self.function.derivative(self.function.x)
        self.x = self.function.x
        self.qx = grad / torch.sqrt(torch.abs(grad) + 1e-3)
        coeffs = natural_cubic_spline_coeffs(self.x, self.qx)
        self.spline = NaturalCubicSpline(coeffs)
        
    def __call__(self, gamma, h=None):
        gamma_x = (self.x[-1] - self.x[0]) * gamma + self.x[0]
        q_gamma = self.spline.evaluate(gamma_x)
        
        return q_gamma
    

class FunctionDistance(ShapeDistanceBase):
    def __init__(self, q, r, k, h=1e-3, sample_type="linear"):
        self.sample_type = sample_type
        super().__init__(q, r, k, h=1e-3)
    def create_point_collection(self, k):
        if self.sample_type=="linear":
            return torch.linspace(0, 1, k).unsqueeze(-1)
        elif self.sample_type=="log":
            return torch.logspace(start=-10, end=0, steps=k).unsqueeze(-1)
            
    def get_determinant(self, network):
        return network.derivative(self.X, self.h)

    def loss_func(self, U, Y):
        r_phi = self.r(Y.squeeze()).squeeze()
        phi_dot = torch.sqrt(U + 1e-8).squeeze()
        error = ((self.Q.squeeze() - phi_dot * r_phi ) ** 2)
        l2_norm = torch.trapezoid(error, x=self.X.squeeze())
        
        return l2_norm

class FunctionsBaseMetric(CurveLayer):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.nvec = torch.arange(1, N + 1, dtype=torch.float)
        self.weights = torch.nn.Parameter(torch.randn(N, 1, requires_grad=True))
        self.weights = torch.nn.init.xavier_uniform_(self.weights)

    @abstractmethod
    def forward(self, x):
        pass 

    @abstractmethod
    def derivative(self, x, h=None):
        pass

    def project(self, eps=1e-6):
        with torch.no_grad():
            scale = (self.Ln*torch.abs(self.weights)).sum()
            if scale > 1.0 - eps:
                self.weights *= (1 - eps) / scale

    def to(self, device):
        self.nvec = self.nvec.to(device)
        return self

class L2Metric(FunctionsBaseMetric):
    def __init__(self, N):
        super().__init__(N)
        self.Ln = (pi/sqrt(2.0))*self.nvec
        self.project()

    def forward(self, x):
        term = torch.sin(pi * self.nvec * x)
        scale = (1/sqrt(2.0))
        return x + (term * scale) @ self.weights 

    def derivative(self, x, h=None):
        term = torch.cos(pi * self.nvec * x)
        scale = (pi*self.nvec)/sqrt(2.0)
        return 1.0 + (term * scale) @ self.weights
    
class PalaisMetric(FunctionsBaseMetric):
    def __init__(self, N):
        super().__init__(N)
        self.weights = torch.nn.Parameter(torch.randn(2*N, 1, requires_grad=True))
        self.weights = torch.nn.init.xavier_uniform_(self.weights)
        self.Ln = sqrt(2.0)
        self.project()

    def forward(self, x):
        t1 = (torch.sin(2.0 * pi * self.nvec * x)/(sqrt(2.0) * pi * self.nvec)) 
        t2 = ((torch.cos(2.0 * pi * self.nvec * x)-1.0)/(sqrt(2.0) * pi * self.nvec)) 
        return x + t1 @ self.weights[:self.N] + t2 @ self.weights[self.N:]

    def derivative(self, x, h=None):
        t1 = torch.cos(2.0 * pi * self.nvec * x)
        t2 = (torch.sin(2.0 * pi * self.nvec * x)) 

        return 1 + ((t1 @ self.weights[:self.N] - t2 @ self.weights[self.N:]) * sqrt(2.0)) 


def get_warping_function(f1 : Function, f2 : Function, **kwargs)->Tuple[Function, CurveReparametrizer , np.ndarray]:
    """Obtain warping function between two functions

    Arguments:
        f1 -- funcshape.functions.Function
        f2 -- funcshape.functions.Function

    Optional:
        n_domain -- number of samples to use in the domain
        domain_type -- type of sampling to use (either "linear" or "log")
        n_restarts -- number of optimization restarts
        n_basis -- number of basis functions to represent the warping manifold tangent space
        n_layers -- number of neural network layers to approximate warping function
        n_iters -- number of optimization iterations
        eps -- threshold for early stopping of optimization

    Returns:
        Tuple of 
            Function -- warped function
            Network -- Optimal network
            error -- best error trace from several restarts
    """
    q1, q2 = SRSF(f1), SRSF(f2)
    # Define loss, optimizer and run reparametrization.
    n_domain = kwargs.get("n_domain", 1024)
    domain_type = kwargs.get("domain_type", "linear")
    loss_func = FunctionDistance(q1, q2, k=n_domain, sample_type=domain_type)

    best_error_value = np.inf
    for _ in range(kwargs.get("n_restarts", 10)):
        basis_type = kwargs.get("basis_type", "palais")
        n_basis = kwargs.get("n_basis", 10)
        if basis_type=="sine":
            basis = SineSeries(n_basis)
        elif basis_type=="L2":
            basis = L2Metric(n_basis)
        elif basis_type=="palais":
            basis = PalaisMetric(n_basis)
        else:
            raise RuntimeError("Basis type %s is not recognised. Should be one of [sine, L2, palais]"%basis_type)

        # Create reparametrization network
        RN = CurveReparametrizer([basis for _ in range(kwargs.get("n_layers", 10))])

        optimizer = optim.LBFGS(RN.parameters(), 
                                lr=kwargs.get("lr", 3e-4), 
                                max_iter=kwargs.get("n_iters", 100), 
                                line_search_fn="strong_wolfe"
                                )
        error = reparametrize(RN, loss_func, optimizer, kwargs.get("n_iters", 100), Logger(0))

        if error[-1]<best_error_value:
            best_error = error
            best_error_value = best_error[-1]
            best_RN = RN
            print("Current best error : %2.4f"%best_error_value)

        if best_error_value<kwargs.get("eps", 1e-2):
            break

    # Get plot data to visualize diffeomorphism
    with torch.no_grad():
        best_RN.detach()
        x = loss_func.create_point_collection(k=n_domain)
        y = best_RN(x)

    return Function(x.squeeze(), y), best_RN, best_error