
from __future__ import annotations

import numpy as np
import networkx as nx

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    torch = None
    Data = None

DEFAULT_ROOT = "./data"


# --------------------------------------------------------------- real datasets
PLANETOID = {"Cora", "CiteSeer", "PubMed"}


def load_dataset(name: str, root: str = DEFAULT_ROOT):
    """Load a benchmark as a PyG Data object."""
    if name in PLANETOID:
        from torch_geometric.datasets import Planetoid
        return Planetoid(root=root, name=name)[0]


# --------------------------------------------------------------- synthetic graphs
def synthetic_graph(kind: str, n: int = 100, seed: int = 0, **kw) -> nx.Graph:
    """Graph families spanning the expansion spectrum.

    kind:
      'regular'   random d-regular expander (kw: d)
      'complete'  complete graph (extreme expander)
      'ba'        Barabasi-Albert, heterogeneous (kw: m)
      'path'      path graph, poor expansion
      'cycle'     cycle graph, satisfies the degree-2 hypothesis
      'sbm'       stochastic block model (kw: blocks, p_in, p_out)
    """
    if kind == "regular":
        return nx.random_regular_graph(kw.get("d", 6), n, seed=seed)
    if kind == "complete":
        return nx.complete_graph(n)
    if kind == "ba":
        return nx.barabasi_albert_graph(n, kw.get("m", 2), seed=seed)
    if kind == "path":
        return nx.path_graph(n)
    if kind == "cycle":
        return nx.cycle_graph(n)
    if kind == "sbm":
        b = kw.get("blocks", 4)
        sizes = [n // b] * b
        p_in, p_out = kw.get("p_in", 0.15), kw.get("p_out", 0.01)
        probs = [[p_in if i == j else p_out for j in range(b)] for i in range(b)]
        return nx.stochastic_block_model(sizes, probs, seed=seed)
    raise ValueError(f"unknown graph kind '{kind}'")


EXPANDER_SUITE = [
    ("3-regular expander", dict(kind="regular", n=100, d=3)),
    ("10-regular expander", dict(kind="regular", n=100, d=10)),
    ("20-regular expander", dict(kind="regular", n=100, d=20)),
    ("complete K_30", dict(kind="complete", n=30)),
    ("path P_100", dict(kind="path", n=100)),
    ("BA(100,1) tree-like", dict(kind="ba", n=100, m=1)),
]


def graph_to_data(G: nx.Graph, features: str = "identity", dim: int = 32,
                  seed: int = 0):
    if Data is None:
        raise ImportError("torch_geometric is required for graph_to_data")
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    ei = []
    for a, b in G.edges():
        ei.append([a, b])
        ei.append([b, a])
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    if features == "identity":
        x = torch.eye(n)
    else:
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, dim, generator=g)
    return Data(x=x, edge_index=edge_index)


# --------------------------------------------------------------- edge handling
def undirected_edges(edge_index) -> list[tuple[int, int]]:
    ei = edge_index.numpy() if hasattr(edge_index, "numpy") else np.asarray(edge_index)
    seen = set()
    for a, b in zip(ei[0], ei[1]):
        a, b = int(a), int(b)
        if a != b:
            seen.add((min(a, b), max(a, b)))
    return sorted(seen)


def split_edges(edges, n_nodes: int, seed: int = 0,
                val_prop: float = 0.05, test_prop: float = 0.10):
    rng = np.random.RandomState(seed)
    E = np.array(edges)
    E = E[rng.permutation(len(E))]
    n_val, n_test = int(val_prop * len(E)), int(test_prop * len(E))
    val, test, train = E[:n_val], E[n_val:n_val + n_test], E[n_val + n_test:]

    pos = set(map(tuple, E.tolist()))
    neg = set()
    target = n_val + n_test + len(train)
    guard = 0
    while len(neg) < target and guard < 500 * target:
        i, j = rng.randint(0, n_nodes, 2)
        guard += 1
        if i == j:
            continue
        e = (min(int(i), int(j)), max(int(i), int(j)))
        if e not in pos:
            neg.add(e)
    N = np.array(sorted(neg))
    N = N[rng.permutation(len(N))]
    return dict(
        train_pos=train, val_pos=val, test_pos=test,
        val_neg=N[:n_val], test_neg=N[n_val:n_val + n_test],
        train_neg=N[n_val + n_test:],
    )
