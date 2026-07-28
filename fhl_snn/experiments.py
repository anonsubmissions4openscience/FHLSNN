
from __future__ import annotations

import time

import numpy as np
import networkx as nx
import scipy.sparse as sp

from . import complexes as cx
from . import data as dta
from . import operators as ops
from .utils import (Results, fmt_pm, loglog_slope, markdown_table, mean_std,
                    semilog_r2, set_seed)

try:
    import torch
    from .models import train_link_prediction
except ImportError:
    torch = None


# ================================================================= preparation
_PREP_CACHE: dict = {}


def prepare(data, split_seed: int = 0, K: int = 16, shift: bool = True,
            use_cache: bool = True):
    """Split edges, build the TRAINING complex, and construct Chebyshev filters."""
    key = (id(data), split_seed, K, shift)
    if use_cache and key in _PREP_CACHE:
        return _PREP_CACHE[key]
    if torch is None:
        raise ImportError("torch is required for prepare()")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = int(data.x.size(0))
    edges = dta.undirected_edges(data.edge_index)
    splits = dta.split_edges(edges, n, seed=split_seed)

    train_edges = sorted(tuple(map(int, e)) for e in splits["train_pos"])
    B1, B2, tris = cx.build_complex(n, train_edges)
    L0, L1, _, _ = cx.hodge_laplacians(B1, B2)

    filt_node = ops.ChebFilter(L0, K=K, shift=shift, device=device)
    filt_edge = ops.ChebFilter(L1, K=K, shift=shift, device=device)
    e2n = ops._sp2torch(cx.incidence_edge_to_node(B1), device)

    X = data.x.to(device)
    src = torch.tensor([e[0] for e in train_edges], device=device)
    dst = torch.tensor([e[1] for e in train_edges], device=device)
    edge_feats = (X[src] + X[dst]).contiguous()

    out = dict(X=X, splits=splits, filt_node=filt_node, filt_edge=filt_edge,
               edge_to_node=e2n, edge_feats=edge_feats, device=device,
               n_nodes=n, train_edges=train_edges, n_triangles=len(tris),
               L0=L0, L1=L1, B1=B1, B2=B2)
    if use_cache:
        _PREP_CACHE[key] = out
    return out


def clear_cache():
    _PREP_CACHE.clear()


# ================================================================= main results
def exp_gamma_sweep(data, gammas=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    seeds=(0, 1, 2), K: int = 16, split_seed: int = 0,
                    use_simplicial: bool = True, epochs: int = 120,
                    store: Results | None = None, tag: str = "",
                    verbose: bool = True):
    """Link-prediction AUC as a function of the fractional exponent gamma."""
    prep = prepare(data, split_seed=split_seed, K=K)
    rows = []
    for g in gammas:
        aucs, aps = [], []
        for s in seeds:
            key = f"gamma|{tag}|{g}|{s}|{K}|{split_seed}|{use_simplicial}"
            if store is not None and store.has(key):
                r = store.get(key)
            else:
                r = train_link_prediction(
                    prep["X"], prep["splits"], prep["filt_node"], prep["filt_edge"],
                    prep["edge_to_node"], prep["edge_feats"], gamma=g, seed=s,
                    epochs=epochs, use_simplicial=use_simplicial,
                    device=prep["device"])
                if store is not None:
                    store.put(key, r)
            aucs.append(r["test_auc"])
            aps.append(r["test_ap"])
        rows.append(dict(gamma=g, auc_mean=mean_std(aucs)[0], auc_std=mean_std(aucs)[1],
                         ap_mean=mean_std(aps)[0], auc_str=fmt_pm(aucs),
                         ap_str=fmt_pm(aps), n_seeds=len(seeds)))
        if verbose:
            print(f"  gamma={g:<4} AUC {rows[-1]['auc_str']}   AP {rows[-1]['ap_str']}")
    return rows


def exp_k_study(data, Ks=(4, 8, 16, 32), gamma: float = 0.5, seeds=(0, 1, 2),
                split_seed: int = 0, epochs: int = 80,
                store: Results | None = None, tag: str = "", verbose: bool = True):
    """Downstream sensitivity to the Chebyshev order K (p3ki Q4)."""
    rows = []
    for K in Ks:
        clear_cache()
        prep = prepare(data, split_seed=split_seed, K=K)
        deficit = ops.row_sum_deficit(gamma, K, prep["filt_edge"].lam_max)
        aucs, times = [], []
        for s in seeds:
            key = f"kstudy|{tag}|{K}|{gamma}|{s}|{split_seed}"
            if store is not None and store.has(key):
                r = store.get(key)
            else:
                t0 = time.perf_counter()
                r = train_link_prediction(
                    prep["X"], prep["splits"], prep["filt_node"], prep["filt_edge"],
                    prep["edge_to_node"], prep["edge_feats"], gamma=gamma, seed=s,
                    epochs=epochs, device=prep["device"])
                r["seconds"] = time.perf_counter() - t0
                if store is not None:
                    store.put(key, r)
            aucs.append(r["test_auc"])
            times.append(r.get("seconds", float("nan")))
        rows.append(dict(K=K, deficit=deficit, auc_str=fmt_pm(aucs),
                         auc_mean=mean_std(aucs)[0], seconds=float(np.mean(times))))
        if verbose:
            print(f"  K={K:<4} deficit={deficit:.4f}  AUC {rows[-1]['auc_str']}  "
                  f"{rows[-1]['seconds']:.0f}s/run")
    clear_cache()
    return rows


# ================================================================= theory checks
def exp_hypothesis_check(datasets: dict, N: int = 1):
    """Do real complexes satisfy the degree-2 hypothesis?  (p3ki Q1)"""
    rows = []
    for name, d in datasets.items():
        n = int(d.x.size(0))
        edges = dta.undirected_edges(d.edge_index)
        _, _, tris = cx.build_complex(n, edges)
        for NN in ([N] if isinstance(N, int) else N):
            h = cx.check_degree_hypothesis(n, edges, tris, N=NN)
            rows.append(dict(dataset=name, N=NN, n_nodes=n, n_edges=len(edges),
                             n_triangles=len(tris),
                             mean_degree=h["mean_degree"],
                             frac_exactly_2=h["frac_exactly_2"],
                             satisfied=h["satisfied"]))
    return rows


def exp_operator_wellposedness(graphs: dict, gamma: float = 0.5):
    """PSD-ness, M-matrix property, kernel, and the lambda_max <= N+1 bound.

    Compares the paper's Eq. (3) operator against the true Hodge down-Laplacian.
    """
    rows = []
    for name, G in graphs.items():
        edges = sorted(tuple(sorted(e)) for e in G.edges())
        n = G.number_of_nodes()
        B1, B2, _ = cx.build_complex(n, edges)
        L_true = (B1.T @ B1).toarray()
        L_eq3 = cx.eq3_down_laplacian(edges)
        for label, L in (("Eq.(3)", L_eq3), ("B1^T B1", L_true)):
            spec = cx.spectrum_summary(L)
            mm = cx.is_m_matrix(L)
            bound = ops.check_lambda_max_bound(L, N=1)
            rows.append(dict(
                graph=name, operator=label, psd=spec["psd"],
                lambda_min=spec["lambda_min"], lambda_2=spec["lambda_2"],
                lambda_max=spec["lambda_max"],
                m_matrix=mm["is_m_matrix"],
                n_pos_offdiag=mm["n_positive_offdiag"],
                norm_L1=mm["norm_L_times_ones"],
                lambda_max_bound_ok=bound["satisfied"],
                violation_factor=bound["violation_factor"]))
    return rows


def exp_eq4_normalisation(graphs: dict, gamma: float = 0.5, p: float = 0.5,
                          N: int = 0):
    """Is P~ stochastic under the fixed rho of Eq. (4)?  (p3ki Q2/Q3)"""
    rho_fixed = 2 * (1 - p) / (N + 1)
    rows = []
    for name, G in graphs.items():
        A = nx.to_numpy_array(G)
        L = np.diag(A.sum(1)) - A
        Lg = ops.exact_fractional(L, gamma)
        lam_max = ops.largest_eigenvalue(L)
        P_fixed = np.eye(len(L)) - rho_fixed * Lg
        P_match = np.eye(len(L)) - (lam_max ** -gamma) * Lg
        rows.append(dict(graph=name, lambda_max=lam_max,
                         fixed_min_entry=float(P_fixed.min()),
                         fixed_stochastic=ops.stochasticity_report(P_fixed)["stochastic"],
                         matched_min_entry=float(P_match.min()),
                         matched_stochastic=ops.stochasticity_report(P_match)["stochastic"]))
    return rows


# ================================================================= Chebyshev
def exp_chebyshev_properties(L, gammas=(0.3, 0.5, 0.7), Ks=(10, 30, 60),
                             shift_variants=(False, True)):
    """Sign pattern, row/column sums, stochasticity vs K and gamma (p3ki Q3)."""
    lam_max = ops.largest_eigenvalue(L)
    rows = []
    for shift in shift_variants:
        for g in gammas:
            for K in Ks:
                Pk = ops.cheb_matrix(L, g, K, lam_max, shift=shift)
                P = np.eye(Pk.shape[0]) - (lam_max ** -g) * Pk
                rep = ops.stochasticity_report(P)
                rows.append(dict(shift=shift, gamma=g, K=K,
                                 min_entry=rep["min_entry"],
                                 n_neg_offdiag=rep["n_negative_offdiag"],
                                 max_rowsum_dev=rep["max_rowsum_dev"],
                                 max_colsum_dev=rep["max_colsum_dev"],
                                 row_col_gap=rep["max_row_colsum_gap"],
                                 rowsum_spread=rep["rowsum_spread"],
                                 norm_inf=rep["norm_inf"],
                                 stochastic=rep["stochastic"]))
    return rows


def exp_chebyshev_error(L, gammas=(0.3, 0.5, 0.7), Ks=(10, 20, 40, 80)):
    """Approximation error vs K: algebraic or geometric decay?  (p3ki Q4, koHV)"""
    lam_max = ops.largest_eigenvalue(L)
    rows = []
    for g in gammas:
        exact = ops.exact_fractional(L, g)
        errs = []
        for K in Ks:
            approx = ops.cheb_matrix(L, g, K, lam_max, shift=False)
            errs.append(np.linalg.norm(approx - exact) / np.linalg.norm(exact))
        slope, r2_ll = loglog_slope(Ks, errs)
        rows.append(dict(gamma=g, Ks=list(Ks), errors=[float(e) for e in errs],
                         loglog_slope=slope, loglog_r2=r2_ll,
                         semilog_r2=semilog_r2(Ks, errs),
                         predicted_slope=-2 * g,
                         deficits=[ops.row_sum_deficit(g, K, lam_max) for K in Ks]))
    return rows


def exp_runtime_comparison(n: int = 1000, gamma: float = 0.5, Ks=(20, 50),
                           seed: int = 1, feature_dims=(1, 64)):
    """Schur-Pade vs Chebyshev.  The speedup depends on the operand shape."""
    G = nx.barabasi_albert_graph(n, 3, seed=seed)
    A = nx.to_numpy_array(G)
    L = np.diag(A.sum(1)) - A
    Ls = sp.csr_matrix(L)
    lam_max = ops.largest_eigenvalue(L)
    rng = np.random.RandomState(0)

    t0 = time.perf_counter()
    ops.schur_pade_fractional(L, gamma)
    t_sp = time.perf_counter() - t0

    rows = [dict(method="Schur-Pade (full matrix)", K=None, operand="n x n",
                 seconds=t_sp, speedup=1.0)]
    for K in Ks:
        c = ops.cheb_coeffs(gamma, K, lam_max)
        Lt = (2.0 / lam_max) * Ls - sp.identity(n, format="csr")

        def cheb_apply(X):
            T0, T1 = X, Lt @ X
            y = (c[0] / 2) * T0 + c[1] * T1
            for k in range(2, K):
                T2 = 2 * (Lt @ T1) - T0
                y = y + c[k] * T2
                T0, T1 = T1, T2
            return y

        for d in feature_dims:
            X = rng.randn(n, d)
            t0 = time.perf_counter()
            for _ in range(5):
                cheb_apply(X)
            t = (time.perf_counter() - t0) / 5
            rows.append(dict(method=f"Chebyshev K={K}", K=K,
                             operand=f"n x {d}", seconds=t, speedup=t_sp / t))
        t0 = time.perf_counter()
        ops.cheb_matrix(L, gamma, K, lam_max)
        t_full = time.perf_counter() - t0
        rows.append(dict(method=f"Chebyshev K={K}", K=K, operand="n x n (full)",
                         seconds=t_full, speedup=t_sp / t_full))
    return rows


# ================================================================= expanders
def exp_expander_mixing(suite=None, gammas=(0.3, 0.5, 0.7, 1.0), eps: float = 0.01):

    suite = suite or dta.EXPANDER_SUITE
    rows = []
    for name, spec in suite:
        G = dta.synthetic_graph(**spec)
        A = nx.to_numpy_array(G)
        L = np.diag(A.sum(1)) - A
        w, V = np.linalg.eigh(L)
        w = np.clip(w, 0, None)
        lam2, lam_max = w[w > 1e-9].min(), w.max()
        base = None
        entries = []
        for g in gammas:
            Lg = V @ np.diag(w ** g) @ V.T
            rho = 1.0 / (lam_max ** g)
            P = np.eye(len(w)) - rho * Lg
            rep = ops.stochasticity_report(P)
            t = ops.mixing_time(P, eps=eps) if rep["stochastic"] else None
            entries.append((g, t, rep["stochastic"]))
            if g == 1.0:
                base = t
        for g, t, ok in entries:
            rows.append(dict(
                graph=name, n=G.number_of_nodes(), m=G.number_of_edges(),
                lambda_2=float(lam2), lambda_max=float(lam_max),
                hypothesis_ok=bool(lam2 < 1), gamma=g,
                t_mix=t, stochastic=ok,
                measured_speedup=(base / t) if (t and base) else None,
                predicted_paper=float(lam2 ** (-(1 - g))),
                predicted_corrected=float((lam2 / lam_max) ** (-(1 - g)))))
    return rows


# ================================================================= 6ZbZ control
def exp_operator_density(L, gammas=(0.3, 0.5, 0.7)):
    """Does L^gamma have more non-zeros than L?  (6ZbZ)"""
    base = ops.density(L)
    rows = [dict(operator="L", **base)]
    for g in gammas:
        Lg = ops.exact_fractional(L, g)
        d = ops.density(Lg)
        rows.append(dict(operator=f"L^{g}", **d,
                         ratio=d["nnz_offdiag"] / max(base["nnz_offdiag"], 1)))
    return rows


def exp_structure_control(data, gamma: float = 0.5, seeds=(0, 1, 2),
                          split_seed: int = 0, epochs: int = 80,
                          verbose: bool = True):
 
    import torch as _t
    import torch.nn as nn
    import torch.nn.functional as Fn
    from sklearn.metrics import roc_auc_score

    n = int(data.x.size(0))
    edges = dta.undirected_edges(data.edge_index)
    splits = dta.split_edges(edges, n, seed=split_seed)
    A = np.zeros((n, n))
    for i, j in splits["train_pos"]:
        A[int(i), int(j)] = A[int(j), int(i)] = 1.0
    L = np.diag(A.sum(1)) - A
    Lg = ops.exact_fractional(L, gamma)
    Lg = (Lg + Lg.T) / 2
    lam_max = ops.largest_eigenvalue(L)
    rho = lam_max ** -gamma

    rng = np.random.RandomState(100 + split_seed)
    iu = np.triu_indices(n, 1)
    vals = Lg[iu].copy()
    Sh = np.zeros((n, n))
    Sh[iu] = vals[rng.permutation(len(vals))]
    Sh = Sh + Sh.T
    np.fill_diagonal(Sh, -Sh.sum(1))

    dens = ops.density(Lg)
    p = dens["density"]
    M = (rng.rand(n, n) < p).astype(float)
    M = np.triu(M, 1)
    M = M + M.T
    total = np.abs(Lg[iu]).sum() * 2
    if M.sum() > 0:
        M *= total / M.sum()
    ER = np.diag(M.sum(1)) - M

    operators = {"FHL": Lg, "SHUFFLED": Sh, "ER": ER}

    class Net(nn.Module):
        def __init__(self, d_in, h=64, o=32):
            super().__init__()
            self.l1, self.l2 = nn.Linear(d_in, h), nn.Linear(h, o)
            self.sc = nn.Sequential(nn.Linear(2 * o, o), nn.ReLU(), nn.Linear(o, 1))
            self.dp = nn.Dropout(0.5)

        def forward(self, X, S):
            h = Fn.relu(self.l1(self.dp(X)))
            h = S @ h
            return S @ self.l2(self.dp(h))

        def dec(self, z, pr):
            a, b = z[pr[:, 0]], z[pr[:, 1]]
            return self.sc(_t.cat([a * b, (a - b).abs()], 1)).squeeze(-1)

    X = data.x
    T = lambda a: _t.tensor(np.asarray(a), dtype=_t.long)
    rows = []
    for label, Lop in operators.items():
        S = _t.tensor(np.eye(n) - rho * Lop, dtype=_t.float32)
        aucs = []
        for s in seeds:
            set_seed(s)
            model = Net(X.shape[1])
            opt = _t.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            best_v, best_t, bad = -1, 0, 0

            def ev(pos, neg):
                model.eval()
                with _t.no_grad():
                    z = model(X, S)
                    sc = _t.cat([model.dec(z, T(pos)), model.dec(z, T(neg))]).numpy()
                return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], sc)

            for ep in range(epochs):
                model.train()
                opt.zero_grad()
                z = model(X, S)
                lo = _t.cat([model.dec(z, T(splits["train_pos"])),
                             model.dec(z, T(splits["train_neg"][:len(splits["train_pos"])]))])
                y = _t.cat([_t.ones(len(splits["train_pos"])),
                            _t.zeros(len(splits["train_pos"]))])
                Fn.binary_cross_entropy_with_logits(lo, y).backward()
                opt.step()
                if ep % 5 == 0:
                    v = ev(splits["val_pos"], splits["val_neg"])
                    if v > best_v:
                        best_v, best_t, bad = v, ev(splits["test_pos"], splits["test_neg"]), 0
                    else:
                        bad += 1
                        if bad > 6:
                            break
            aucs.append(best_t)
        rows.append(dict(operator=label, auc_str=fmt_pm(aucs),
                         auc_mean=mean_std(aucs)[0], auc_std=mean_std(aucs)[1]))
        if verbose:
            print(f"  {label:9s} AUC {rows[-1]['auc_str']}")
    return rows, dens


# ================================================================= cost
def exp_preprocessing_cost(datasets: dict, repeat: int = 3):
    """Wall-clock for simplicial complex construction (p3ki Q5)."""
    rows = []
    for name, d in datasets.items():
        n = int(d.x.size(0))
        edges = dta.undirected_edges(d.edge_index)
        rows.append(dict(dataset=name, **cx.preprocessing_cost(n, edges, repeat=repeat)))
    return rows
