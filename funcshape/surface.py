import torch
import numpy as np 
import gpytorch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings 
from funcshape.diffeomorphism import Diffeomorphism2D

class Surface:
    """Torch-compatible surface class. Constructed from a tuple of functions,
    mapping tensor of dim (..., 2) to R^len(tuple)"""

    def __init__(self, component_function_tuple, **kwargs):
        super().__init__(**kwargs)
        self.S = tuple(component_function_tuple)
        self.dim = len(self.S)

    def __call__(self, X):
        return torch.cat([ci(X).unsqueeze(dim=-1) for ci in self.S], dim=-1)

    def partial_derivative(self, X, component, h):
        if h is None:
            raise ValueError(
                f"{self.__class__} has not implemented partial"
                + f" derivatives. Needs variable h={h} to be float"
                + f" to enable finite difference approximation."
            )
        H = torch.zeros_like(X, device=X.device)
        H[..., component] = h
        return (0.5 / h) * torch.cat(
            [(ci(X + H) - ci(X - H)).unsqueeze(dim=-1) for ci in self.S], dim=-1
        )

    def volume_factor(self, X, h):
        return torch.norm(self.normal_vector(X, h), dim=-1, keepdim=True)

    def normal_vector(self, X, h):
        dfx = self.partial_derivative(X, 0, h)
        dfy = self.partial_derivative(X, 1, h)
        return torch.cross(dfx, dfy, dim=-1)

    def compose(self, f):
        return Surface(
            (
                lambda x: self.S[0](f(x)),
                lambda x: self.S[1](f(x)),
                lambda x: self.S[2](f(x)),
            )
        )


class ComposedSurface(Surface):
    def __init__(self, surf: Surface, diffeomorphism: Diffeomorphism2D):
        self.s = surf
        self.diffeo = diffeomorphism
        self.dim = surf.dim

    def __call__(self, X):
        return self.s(self.diffeo(X))

    def normal_vector(self, X, h):
        J = self.diffeo.jacobian_determinant(X, h)
        n = self.s.normal_vector(self.diffeo(X), h)
        return J * n

def check_scaled_to_unit_interval(arr, tol=1e-2):
    """
    Check whether all values in 'arr' are scaled to [0, 1].
    Raises a warning if not.

    Parameters
    ----------
    arr : array-like
        Input array to check.
    tol : float, optional
        Numerical tolerance to account for floating-point errors.

    Returns
    -------
    bool
        True if array is within [0, 1] (within tolerance), False otherwise.
    """
    arr = np.asarray(arr)

    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if min_val < -tol or max_val > 1 + tol:
        warnings.warn(
            f"Array values are not within [0, 1]. "
            f"Found min={min_val:.4f}, max={max_val:.4f}.",
            UserWarning
        )
        return False

    return True


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NormalizedGPRegressor:
    def __init__(self, training_iter=200, lr=0.1, device='cpu'):
        self.training_iter = training_iter
        self.lr = lr
        self.device = device

        self.x_scaler = MinMaxScaler()
        self.y_scaler = StandardScaler()

        self.model = None
        self.likelihood = None

    def fit(self, X, y):
        # Ensure numpy arrays
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        # Scale input and target
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y).ravel()

        # Convert to torch tensors
        train_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        # Initialize model and likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = GPRegressionModel(train_x, train_y, self.likelihood).to(self.device)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes likelihood parameters
        ], lr=self.lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X_test, return_std=True):
        X_test = np.asarray(X_test)
        check_scaled_to_unit_interval(X_test)
        test_x = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(test_x))
            mean = preds.mean.cpu().numpy()
            std = preds.stddev.cpu().numpy()

        # Inverse-transform to original target scale
        mean_original = self.y_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
        std_original = std * self.y_scaler.scale_[0]

        return (mean_original, std_original) if return_std else mean_original


class PointCloudSurface(Surface):
    def __init__(self, points, uv=None):
        """
        points: (N, 3) array of raw point cloud
        uv: optional (N, 2) parameterization
        """
        self.points = np.asarray(points)

        if uv is None:
            uv = self.points[:,[0,1]]

        self.uv = uv

        # Step 2: Fit Gaussian processes for mapping UV -> XYZ
        self.pipeline = NormalizedGPRegressor(training_iter=150, lr=0.05)
        self.pipeline.fit(self.uv, self.points[:, 2])

        # Pass callable mapping into Surface base class
        super().__init__(
            (
                lambda x: torch.tensor(self.pipeline.x_scaler.inverse_transform(x.detach().cpu().numpy())[...,0], dtype=torch.float32),
                lambda x: torch.tensor(self.pipeline.x_scaler.inverse_transform(x.detach().cpu().numpy())[...,1], dtype=torch.float32),
                lambda x: torch.tensor(self.pipeline.predict(x.detach().cpu().numpy())[0], dtype=torch.float32)
            )
        )