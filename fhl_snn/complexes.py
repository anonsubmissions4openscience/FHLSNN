"""Simplicial complexes, boundary operators and Hodge Laplacians.

Conventions (satisfying B1 @ B2 == 0):
  B1 : |V| x |E|   vertex-edge,   edge (i<j) has -1 at i, +1 at j
  B2 : |E| x |F|   edge-triangle, triangle (i<j<k) has +1,+1,-1 on (i,j),(j,k),(i,k)
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------- construction
def build_complex(n_nodes: int, edges):
    """Return (B1, B2, triangles). `edges` is a sorted list of (i, j), i < j."""
    edges = [tuple(sorted(map(int, e))) for e in edges]
    edges = sorted(set(edges))
    eidx = {e: k for k, e in enumerate(edges)}
    m = len(edges)

    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(edges):
        rows += [i, j]
        cols += [k, k]
        vals += [-1.0, 1.0]
    B1 = sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, m))

    adj = [set() for _ in range(n_nodes)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    tris = sorted({tuple(sorted((i, j, k)))
                   for (i, j) in edges for k in (adj[i] & adj[j])})

    if tris:
        rows, cols, vals = [], [], []
        for f, (i, j, k) in enumerate(tris):
            rows += [eidx[(i, j)], eidx[(j, k)], eidx[(i, k)]]
            cols += [f, f, f]
            vals += [1.0, 1.0, -1.0]
        B2 = sp.csr_matrix((vals, (rows, cols)), shape=(m, len(tris)))
    else:
        B2 = sp.csr_matrix((m, 1))
    return B1, B2, tris


def hodge_laplacians(B1, B2):
    """L0 (nodes), L1 (edges) = L1_down + L1_up.  Both PSD by construction."""
    L0 = (B1 @ B1.T).tocsr()
    L1_down = (B1.T @ B1).tocsr()
    L1_up = (B2 @ B2.T).tocsr()
    return L0, (L1_down + L1_up).tocsr(), L1_down, L1_up


def eq3_down_laplacian(edges):
    """The paper's Eq. (3) operator for N=1 (unsigned, degree-2 form).

    L f(s) = f(s) - 0.5 * sum_{s' sharing a vertex with s} f(s')

    Only PSD when every vertex has degree 2; see `check_degree_hypothesis`.
    """
    edges = [tuple(sorted(map(int, e))) for e in sorted(set(map(tuple, edges)))]
    m = len(edges)
    L = np.zeros((m, m))
    for a in range(m):
        L[a, a] = 1.0
        sa = set(edges[a])
        for b in range(m):
            if a != b and sa & set(edges[b]):
                L[a, b] = -0.5
    return L


def incidence_edge_to_node(B1):
    """Row-normalised |B1|: averages incident edge signals onto nodes."""
    A = abs(B1).tocsr()
    rs = np.asarray(A.sum(1)).ravel()
    rs[rs == 0] = 1.0
    return (sp.diags(1.0 / rs) @ A).tocsr()


# --------------------------------------------------------------- diagnostics
def check_degree_hypothesis(n_nodes, edges, tris=None, N: int = 1) -> dict:
    """Theorem 3.4/3.6 require every (N-1)-simplex to have degree exactly 2.

    N=1: every vertex lies in exactly 2 edges.
    N=2: every edge lies in exactly 2 triangles.
    """
    edges = [tuple(sorted(map(int, e))) for e in edges]
    if N == 1:
        deg = np.zeros(n_nodes)
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        deg = deg[deg > 0]
    elif N == 2:
        if tris is None:
            _, _, tris = build_complex(n_nodes, edges)
        cnt = {e: 0 for e in edges}
        for i, j, k in tris:
            for e in ((i, j), (j, k), (i, k)):
                if e in cnt:
                    cnt[e] += 1
        deg = np.array(list(cnt.values()), dtype=float)
    else:
        raise ValueError("N must be 1 or 2")
    return dict(
        N=N,
        mean_degree=float(deg.mean()) if len(deg) else 0.0,
        frac_exactly_2=float((deg == 2).mean()) if len(deg) else 0.0,
        satisfied=bool(len(deg) and np.all(deg == 2)),
    )


def spectrum_summary(L) -> dict:
    """lambda_2 (smallest non-zero), lambda_max, PSD flag, kernel dimension."""
    dense = L.toarray() if sp.issparse(L) else np.asarray(L)
    w = np.linalg.eigvalsh((dense + dense.T) / 2)
    nz = w[w > 1e-9]
    return dict(
        lambda_min=float(w.min()),
        lambda_2=float(nz.min()) if len(nz) else float("nan"),
        lambda_max=float(w.max()),
        psd=bool(w.min() > -1e-9),
        kernel_dim=int((np.abs(w) <= 1e-9).sum()),
    )


def is_m_matrix(L, tol: float = 1e-9) -> dict:
    """M-matrix requires non-positive off-diagonal entries."""
    dense = L.toarray() if sp.issparse(L) else np.asarray(L)
    off = dense - np.diag(np.diag(dense))
    n_pos = int((off > tol).sum())
    ones = np.ones(dense.shape[0])
    return dict(
        n_positive_offdiag=n_pos,
        is_m_matrix=bool(n_pos == 0),
        norm_L_times_ones=float(np.linalg.norm(dense @ ones)),
    )


def preprocessing_cost(n_nodes, edges, repeat: int = 1) -> dict:
    """Wall-clock for complex construction (reviewer p3ki, question 5)."""
    t0 = time.perf_counter()
    for _ in range(repeat):
        B1, B2, tris = build_complex(n_nodes, edges)
    t_complex = (time.perf_counter() - t0) / repeat

    t0 = time.perf_counter()
    for _ in range(repeat):
        hodge_laplacians(B1, B2)
    t_lap = (time.perf_counter() - t0) / repeat

    L0, L1, _, _ = hodge_laplacians(B1, B2)
    return dict(
        n_nodes=n_nodes, n_edges=len(edges), n_triangles=len(tris),
        t_complex_s=t_complex, t_laplacian_s=t_lap,
        t_total_s=t_complex + t_lap,
        nnz_L0=int(L0.nnz), nnz_L1=int(L1.nnz),
    )
