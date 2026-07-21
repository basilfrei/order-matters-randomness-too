import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse

from torch.nn.utils import parameters_to_vector, vector_to_parameters


# LEAST SQUARES PROBLEM
class LeastSquares:
    def __init__(self, A, b, reg=0.0):
        self.A = A
        self.b = b
        self.reg = reg
        self.n, self.d = A.shape

    def loss(self, x):
        residual = self.A @ x - self.b
        data_loss = 0.5 * torch.mean(residual ** 2)
        reg_loss = 0.5 * self.reg * torch.sum(x ** 2)

        return data_loss + reg_loss

    def gradient(self, x):
        residual = self.A @ x - self.b
        data_grad = self.A.T @ residual / self.n
        reg_grad = self.reg * x

        return data_grad + reg_grad

    def stochastic_gradient(self, x, idx, normalization=None):
        # normalization defaults to the actual batch length; permutation-based
        # optimizers pass a constant so an incomplete last batch is not overweighted
        if normalization is None:
            normalization = len(idx)
        A_batch = self.A[idx]
        b_batch = self.b[idx]
        residual_batch = A_batch @ x - b_batch
        data_grad = A_batch.T @ residual_batch / normalization
        reg_grad = self.reg * x

        return data_grad + reg_grad

    def compute_optimum_analytically(self):
        I = torch.eye(self.d)
        H = self.A.T @ self.A + self.n * self.reg * I
        rhs = self.A.T @ self.b
        x_opt = torch.linalg.solve(H, rhs)
        f_opt = self.loss(x_opt).item()

        return x_opt, f_opt


# for smoke tests
def create_synthetic_least_squares(n, d, noise_std=0.1, seed=42):
    torch.manual_seed(seed)

    A = torch.randn(n, d)
    x_star = torch.randn(d)

    b = A @ x_star + noise_std * torch.randn(n)

    return A, b, x_star


# LOGISTIC REGRESSION PROBLEM (labels in {-1, +1})
class LogisticRegression:
    def __init__(self, A, b, reg=0.0):
        self.A = A
        self.b = b
        self.reg = reg
        self.n, self.d = A.shape

        self._sparse = scipy.sparse.issparse(A)   # sparse support for real-sim / RCV1

    # Helper used only when A is a scipy sparse matrix
    def _Ax(self, A_sub, x):
        result = np.asarray(A_sub @ x.detach().cpu().numpy()).ravel()
        return torch.tensor(result, dtype=x.dtype, device=x.device)

    # Helper used only when A is a scipy sparse matrix
    def _ATv(self, A_sub, v):
        result = np.asarray(A_sub.T @ v.detach().cpu().numpy()).ravel()
        return torch.tensor(result, dtype=v.dtype, device=v.device)

    def matvec(self, A, x):
        if self._sparse:
            return self._Ax(A, x)
        return A @ x

    def tmatvec(self, A, v):
        if self._sparse:
            return self._ATv(A, v)
        return A.T @ v

    def loss(self, x):
        logits = self.matvec(self.A, x)
        data_loss = torch.mean(F.softplus(-self.b * logits))
        reg_loss = 0.5 * self.reg * torch.sum(x ** 2)

        return data_loss + reg_loss

    def gradient(self, x):
        logits = self.matvec(self.A, x)
        weights = torch.sigmoid(-self.b * logits)
        data_grad = -self.tmatvec(self.A, self.b * weights) / self.n
        reg_grad = self.reg * x

        return data_grad + reg_grad

    def stochastic_gradient(self, x, idx, normalization=None):
        # normalization defaults to the actual batch length; permutation-based
        # optimizers pass a constant so an incomplete last batch is not overweighted
        if normalization is None:
            normalization = len(idx)

        if self._sparse:
            key = idx.numpy() if torch.is_tensor(idx) else idx
            A_batch = self.A[key]
        else:
            A_batch = self.A[idx]

        b_batch = self.b[idx]

        logits = self.matvec(A_batch, x)
        weights = torch.sigmoid(-b_batch * logits)
        data_grad = -self.tmatvec(A_batch, b_batch * weights) / normalization
        reg_grad = self.reg * x

        return data_grad + reg_grad


# for smoke tests (labels in {-1, +1})
def create_synthetic_logistic_regression(n, d, seed=42):
    torch.manual_seed(seed)

    A = torch.randn(n, d)
    x_star = torch.randn(d)

    logits = A @ x_star
    probs = torch.sigmoid(logits)

    # sample labels in {0, 1}
    y = torch.bernoulli(probs)
    # convert them to {-1, 1}
    b = 2 * y - 1

    return A, b, x_star


# NEURAL NETWORK OBJECTIVES
#
# Shared base class + objective-specific subclasses; see FlatParameterNN's
# docstring for the flat-parameter interface every subclass relies on.


class FlatParameterNN(nn.Module):
    """
    Base class for the ReLU neural-network objectives.

    Owns all parameter management shared across the regression/classification and
    dense/sparse variants, and exposes the flat parameter-vector interface the
    optimizers rely on: the network's weights are treated as a single 1-D vector
    ``x`` that is read with ``get_params()`` and written back with
    ``_set_params(x)``. Subclasses implement only what is specific to them — the
    loss and the data handling (dense vs. sparse, full-batch vs. minibatch) —
    by overriding ``loss`` / ``gradient`` / ``stochastic_gradient``.

    Architecture (built by ``_build_network``):
        d_in -> [Linear -> ReLU] for each hidden width -> Linear(d_out)

    ``hidden_dim`` may be either an int (a uniform width repeated
    ``num_hidden_layers`` times) or a list/tuple of per-layer widths (e.g.
    ``[128, 64]`` for a 784 -> 128 -> 64 -> d_out funnel, in which case
    ``num_hidden_layers`` is implied by the list length). Defaults to 128.
    """

    def __init__(self, d_in, d_out, num_hidden_layers=2, hidden_dim=None,
                 reg=0.0, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.reg = reg
        self.d = d_in
        self.d_out = d_out
        self.hidden_dims = self._resolve_hidden_dims(hidden_dim, num_hidden_layers)
        self._build_network(d_in, d_out, self.hidden_dims)
        self.network.to(self.device)

    @staticmethod
    def _resolve_hidden_dims(hidden_dim, num_hidden_layers):
        """Normalize ``hidden_dim`` into an explicit per-layer list of widths.

        None -> [128] * num_hidden_layers;
        int -> [hidden_dim] * num_hidden_layers (uniform);
        list/tuple -> used as-is (num_hidden_layers is then ignored).
        """
        if hidden_dim is None:
            hidden_dim = 128
        if isinstance(hidden_dim, int):
            return [hidden_dim] * num_hidden_layers
        return list(hidden_dim)

    def _build_network(self, d_in, d_out, hidden_dims):
        """Build the shared ReLU NN from a list of hidden widths, store as ``self.network``."""
        layers, prev = [], d_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.network = nn.Sequential(*layers)

    def _set_params(self, x):
        """Copy the flat parameter vector ``x`` into the network in place."""
        with torch.no_grad():
            vector_to_parameters(x, self.network.parameters())

    def get_params(self):
        """Return the network parameters flattened into a single 1-D vector."""
        return parameters_to_vector(self.network.parameters()).detach()

    def _flat_grad(self):
        """Flatten every parameter's current ``.grad`` into a single 1-D vector."""
        return torch.cat([
            p.grad.detach().reshape(-1)
            if p.grad is not None
            else torch.zeros_like(p).reshape(-1)
            for p in self.network.parameters()
        ])


# NEURAL NETWORK REGRESSION OBJECTIVE
class SimpleRegressionNN(FlatParameterNN):
    """
    ReLU network with MSE loss, for regression tasks (synthetic NN, California).

    Inherits the flat-parameter interface from ``FlatParameterNN`` and only
    adds the MSE objective. ``b`` may be 1-D ``(n,)`` — it is reshaped to
    ``(n, d_out)`` automatically. ``hidden_dim`` defaults to 128 and
    ``num_hidden_layers`` sets the number of hidden ReLU layers.
    """

    def __init__(self, A, b, num_hidden_layers=2, hidden_dim=None, reg=0.0, device="cpu"):
        b = b.unsqueeze(1) if b.dim() == 1 else b
        super().__init__(d_in=A.shape[1],
                         d_out=b.shape[1],
                         num_hidden_layers=num_hidden_layers,
                         hidden_dim=hidden_dim,
                         reg=reg,
                         device=device
                        )
        self.A = A.to(self.device)
        self.b = b.to(self.device)
        self.n = A.shape[0]

    def loss(self, x):
        self._set_params(x)
        with torch.no_grad():  # no backward graph needed for loss reading
            data_loss = F.mse_loss(self.network(self.A), self.b)

        reg_loss = 0.5 * self.reg * torch.sum(x ** 2)

        return data_loss + reg_loss

    def gradient(self, x):
        self._set_params(x)
        self.network.zero_grad()
        # compute the gradient of the MSE loss wrt to the network parameters
        F.mse_loss(self.network(self.A), self.b).backward()

        data_grad = self._flat_grad()
        reg_grad = self.reg * x

        return data_grad + reg_grad

    def stochastic_gradient(self, x, idx, normalization=None):
        self._set_params(x)
        self.network.zero_grad()
        idx = idx.to(self.device)
        loss = F.mse_loss(self.network(self.A[idx]), self.b[idx])

        if normalization is not None:
            # rescale the mean-reduction loss so an incomplete last batch is
            # weighted by the constant normalization instead of its own length
            loss = loss * (len(idx) / normalization)

        # compute the gradient of the batch MSE loss wrt to the network parameters
        loss.backward()

        data_grad = self._flat_grad()
        reg_grad = self.reg * x

        return data_grad + reg_grad


# NEURAL NETWORK CLASSIFICATION OBJECTIVE (dense input)
class SimpleClassificationNN(FlatParameterNN):
    """
    ReLU classifier with softmax + cross-entropy loss, for dense classification
    tasks (MNIST, Fashion-MNIST).

    Inherits the flat-parameter interface from ``FlatParameterNN`` and only adds
    the cross-entropy objective. The final layer outputs raw logits; the softmax
    is applied inside ``F.cross_entropy``. ``hidden_dim`` defaults to 128 and
    ``num_hidden_layers`` sets the number of hidden ReLU layers.

    b must be a 1-D tensor of integer class labels in {0, ..., num_classes-1}.
    """

    def __init__(self, A, b, num_classes, num_hidden_layers=2, hidden_dim=None,
                 reg=0.0, device="cpu"):
        super().__init__(d_in=A.shape[1],
                         d_out=num_classes,
                         num_hidden_layers=num_hidden_layers,
                         hidden_dim=hidden_dim,
                         reg=reg,
                         device=device
                        )
        self.A = A.to(self.device)
        self.b = b.long().to(self.device)
        self.n = A.shape[0]
        self.num_classes = num_classes

    def loss(self, x):
        self._set_params(x)
        with torch.no_grad():  # no backward graph needed for loss reading
            data_loss = F.cross_entropy(self.network(self.A), self.b)

        reg_loss = 0.5 * self.reg * torch.sum(x ** 2)

        return data_loss + reg_loss

    def gradient(self, x):
        self._set_params(x)
        self.network.zero_grad()
        # compute the gradient of the cross-entropy loss wrt to the network parameters
        F.cross_entropy(self.network(self.A), self.b).backward()

        data_grad = self._flat_grad()
        reg_grad = self.reg * x

        return data_grad + reg_grad

    def stochastic_gradient(self, x, idx, normalization=None):
        self._set_params(x)
        self.network.zero_grad()
        idx = idx.to(self.device)
        loss = F.cross_entropy(self.network(self.A[idx]), self.b[idx])

        if normalization is not None:
            # rescale the mean-reduction loss so an incomplete last batch is
            # weighted by the constant normalization instead of its own length
            loss = loss * (len(idx) / normalization)

        # compute the gradient of the batch cross-entropy loss wrt to the network parameters
        loss.backward()

        data_grad = self._flat_grad()
        reg_grad = self.reg * x

        return data_grad + reg_grad


# for smoke tests
def create_synthetic_simple_NN(n, d, d_out=1, num_layers=2, hidden_dim=20, noise_std=0.1, seed=42):
    """Regression targets generated by a fixed random ReLU network (the same
    architecture SimpleRegressionNN would build), so a SimpleRegressionNN can
    be fit against a known, learnable ground truth."""
    torch.manual_seed(seed)

    layers, prev = [], d
    for _ in range(num_layers):
        layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
        prev = hidden_dim
    layers.append(nn.Linear(prev, d_out))
    target_net = nn.Sequential(*layers)

    A = torch.randn(n, d)
    with torch.no_grad():
        b = target_net(A).squeeze(-1) + noise_std * torch.randn(n)

    return A, b
