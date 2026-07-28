
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigsh

try:
    import torch
except ImportError:
    torch = None


# --------------------------------------------------------------- exact
def exact_fractional(L, gamma: float) -> np.ndarray:
    """L^gamma via eigendecomposition.  Requires L symmetric PSD."""
    dense = L.toarray() if sp.issparse(L) else np.asarray(L, dtype=float)
    dense = (dense + dense.T) / 2
    w, V = np.linalg.eigh(dense)
    if w.min() < -1e-8:
        raise ValueError(
            f"operator is not PSD (min eigenvalue {w.min():.4g}); "
            "L^gamma is not real-valued. Definition 3.1 does not apply here."
        )
    w = np.clip(w, 0.0, None)
    return V @ np.diag(w ** gamma) @ V.T


def schur_pade_fractional(L, gamma: float) -> np.ndarray:
    """Reference implementation used for the runtime comparison."""
    dense = L.toarray() if sp.issparse(L) else np.asarray(L, dtype=float)
    return np.real(fractional_matrix_power(dense, gamma))


def largest_eigenvalue(L) -> float:
    if sp.issparse(L):
        try:
            return float(eigsh(L, k=1, which="LM", return_eigenvectors=False)[0])
        except Exception:
            return float(abs(L).sum(axis=1).max())
    return float(np.linalg.eigvalsh((L + L.T) / 2).max())


# --------------------------------------------------------------- Chebyshev
def cheb_coeffs(gamma: float, K: int, lam_max: float) -> np.ndarray:
    """Chebyshev coefficients of t^gamma on [0, lam_max] (paper Algorithm 1)."""
    l = np.arange(K)
    nodes = np.cos(np.pi * (l + 0.5) / K)
    lam = (nodes + 1) * lam_max / 2.0
    f = lam ** gamma
    return np.array([(2.0 / K) * np.sum(f * np.cos(np.pi * k * (l + 0.5) / K))
                     for k in range(K)])


def cheb_at_zero(gamma: float, K: int, lam_max: float) -> float:
    """p_K(0).  Rescaled argument is -1, and T_k(-1) = (-1)^k."""
    c = cheb_coeffs(gamma, K, lam_max)
    return float(c[0] / 2 + sum(c[k] * (-1) ** k for k in range(1, K)))


def row_sum_deficit(gamma: float, K: int, lam_max: float) -> float:
    """Uniform row-sum shortfall  rho * p_K(0)  with rho = lam_max^-gamma.

    Independent of the graph: p_K(0) scales as lam_max^gamma, cancelling rho.
    """
    return cheb_at_zero(gamma, K, lam_max) / (lam_max ** gamma)


def cheb_matrix(L, gamma: float, K: int, lam_max: float | None = None,
                shift: bool = False) -> np.ndarray:
    """Dense p_K(L) (or q_K(L) if shift). For diagnostics on small operators."""
    dense = L.toarray() if sp.issparse(L) else np.asarray(L, dtype=float)
    n = dense.shape[0]
    lam_max = largest_eigenvalue(dense) if lam_max is None else lam_max
    c = cheb_coeffs(gamma, K, lam_max)
    Lt = (2.0 / lam_max) * dense - np.eye(n)
    T0, T1 = np.eye(n), Lt.copy()
    out = (c[0] / 2) * T0 + c[1] * T1
    for k in range(2, K):
        T2 = 2 * Lt @ T1 - T0
        out = out + c[k] * T2
        T0, T1 = T1, T2
    if shift:
        out = out - cheb_at_zero(gamma, K, lam_max) * np.eye(n)
    return out


class ChebFilter:
    """Applies P_gamma = I - rho L^gamma with sparse matvecs, O(K * nnz)."""

    def __init__(self, L, K: int = 24, lam_max: float | None = None,
                 shift: bool = True, device=None):
        if torch is None:
            raise ImportError("torch is required for ChebFilter")
        self.n = L.shape[0]
        self.K = K
        self.shift = shift
        self.lam_max = largest_eigenvalue(L) if lam_max is None else lam_max
        if self.lam_max <= 0:
            self.lam_max = 1.0
        self.device = device or torch.device("cpu")
        Lt = (2.0 / self.lam_max) * sp.csr_matrix(L) - sp.identity(self.n, format="csr")
        self.Lt = _sp2torch(Lt, self.device)
        self._cache: dict[float, torch.Tensor] = {}

    def _coeffs(self, gamma: float):
        if gamma not in self._cache:
            c = cheb_coeffs(gamma, self.K, self.lam_max)
            self._cache[gamma] = torch.tensor(c, dtype=torch.float32,
                                              device=self.device)
        return self._cache[gamma]

    def apply_power(self, X, gamma: float):
        """Approximate L^gamma X."""
        c = self._coeffs(gamma)
        T0 = X
        T1 = torch.sparse.mm(self.Lt, X)
        y = (c[0] / 2.0) * T0 + c[1] * T1
        for k in range(2, self.K):
            T2 = 2.0 * torch.sparse.mm(self.Lt, T1) - T0
            y = y + c[k] * T2
            T0, T1 = T1, T2
        if self.shift:
            y = y - cheb_at_zero(gamma, self.K, self.lam_max) * X
        return y

    def apply(self, X, gamma: float):
        """P_gamma X = X - rho L^gamma X."""
        rho = 1.0 / (self.lam_max ** gamma)
        return X - rho * self.apply_power(X, gamma)


def _sp2torch(A, device):
    A = sp.coo_matrix(A)
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce().to(device)


# --------------------------------------------------------------- diagnostics
def stochasticity_report(P, tol: float = 1e-9) -> dict:
    """Non-negativity, sign pattern, row/column sums, operator norms."""
    P = np.asarray(P)
    n = P.shape[0]
    off = P - np.diag(np.diag(P))
    rs, cs = P.sum(1), P.sum(0)
    return dict(
        min_entry=float(P.min()),
        nonnegative=bool(P.min() >= -tol),
        n_negative_offdiag=int((off < -tol).sum()),
        max_asymmetry=float(np.abs(P - P.T).max()),
        max_rowsum_dev=float(np.abs(rs - 1).max()),
        max_colsum_dev=float(np.abs(cs - 1).max()),
        rowsum_spread=float(rs.max() - rs.min()),
        max_row_colsum_gap=float(np.abs(rs - cs).max()),
        norm_inf=float(np.abs(P).sum(1).max()),
        norm_1=float(np.abs(P).sum(0).max()),
        stochastic=bool(P.min() >= -tol and np.abs(rs - 1).max() < 1e-8),
    )


def check_lambda_max_bound(L, N: int = 1) -> dict:
    """Theorem 3.4's proof uses lambda_max <= N+1.  Test it."""
    lam_max = largest_eigenvalue(L)
    return dict(lambda_max=lam_max, bound=N + 1,
                satisfied=bool(lam_max <= N + 1),
                violation_factor=float(lam_max / (N + 1)))


def mixing_time(P, eps: float = 0.01, t_max: int = 200000) -> int | None:
    """Measured mixing time to total-variation distance eps from uniform."""
    P = np.asarray(P)
    n = P.shape[0]
    pi = np.ones(n) / n
    p = np.zeros(n)
    p[0] = 1.0
    for t in range(1, t_max + 1):
        p = p @ P
        if 0.5 * np.abs(p - pi).sum() <= eps:
            return t
    return None


def density(L, tol: float = 1e-8) -> dict:
    """Off-diagonal non-zero count and density."""
    dense = L.toarray() if sp.issparse(L) else np.asarray(L)
    n = dense.shape[0]
    nnz = int((np.abs(dense) > tol).sum() - np.count_nonzero(np.abs(np.diag(dense)) > tol))
    return dict(n=n, nnz_offdiag=nnz, density=nnz / (n * (n - 1)))
