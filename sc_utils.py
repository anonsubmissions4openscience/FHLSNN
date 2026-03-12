import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.optim as optim


N_GROUND_TRUTH = 10000

class sc_fromPC:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        self.n_points = len(self.points)
        self.vertices = None
        self.edges = None
        self.triangles = None
        self.edge_to_idx = {}
        self.tri_to_idx = {}
        self.B1 = None
        self.B2 = None
        self.adjacency_0 = None
        self.incidence_1 = None
        self.incidence_2 = None
        self.L0 = None
        self.L1 = None
        self.L2 = None

    def build_rips(self, max_edge_length=None, max_dimension=2):
        if max_edge_length is None:
            tree = cKDTree(self.points)
            dists, _ = tree.query(self.points, k=2)
            max_edge_length = float(np.median(dists[:, 1]) * 2.0)
        rips = gudhi.RipsComplex(
            points=self.points.tolist(), max_edge_length=max_edge_length
        )
        st = rips.create_simplex_tree(max_dimension=max_dimension)
        self._extract_from_simplex_tree(st)
        return self

    def _extract_from_simplex_tree(self, simplex_tree):
        verts, edges, tris = [], [], []
        for simplex, _ in simplex_tree.get_simplices():
            dim = len(simplex) - 1
            if dim == 0:
                verts.append(simplex[0])
            elif dim == 1:
                edges.append(tuple(sorted(simplex)))
            elif dim == 2:
                tris.append(tuple(sorted(simplex)))

        self.vertices = np.array(sorted(set(verts)))
        self.edges = np.array(sorted(edges)) if edges else np.empty((0, 2), dtype=int)
        self.triangles = np.array(sorted(tris)) if tris else np.empty((0, 3), dtype=int)
        self.edge_to_idx = {tuple(e): i for i, e in enumerate(self.edges)}
        self.tri_to_idx = {tuple(t): i for i, t in enumerate(self.triangles)}
        self._build_boundary_operators()
        self._build_adjacency_incidence()
        self._build_hodge_laplacians()

    def _build_boundary_operators(self):
        N = self.n_points
        E = len(self.edges)
        T = len(self.triangles)

        if E > 0:
            rows, cols, vals = [], [], []
            for k, (i, j) in enumerate(self.edges):
                rows.extend([i, j])
                cols.extend([k, k])
                vals.extend([-1, 1])
            self.B1 = sparse.csr_matrix((vals, (rows, cols)), shape=(N, E))
        else:
            self.B1 = sparse.csr_matrix((N, 0))

        if T > 0 and E > 0:
            rows, cols, vals = [], [], []
            for m, (i, j, k) in enumerate(self.triangles):
                e_jk = self.edge_to_idx.get((j, k))
                e_ik = self.edge_to_idx.get((i, k))
                e_ij = self.edge_to_idx.get((i, j))
                if e_jk is not None:
                    rows.append(e_jk); cols.append(m); vals.append(1)
                if e_ik is not None:
                    rows.append(e_ik); cols.append(m); vals.append(-1)
                if e_ij is not None:
                    rows.append(e_ij); cols.append(m); vals.append(1)
            self.B2 = sparse.csr_matrix((vals, (rows, cols)), shape=(E, T))
        else:
            self.B2 = sparse.csr_matrix((E, 0))

    def _build_adjacency_incidence(self):
        N = self.n_points
        E = len(self.edges)
        T = len(self.triangles)

        if E > 0:
            rows = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
            cols = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
            vals = np.ones(2 * E)
            self.adjacency_0 = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
        else:
            self.adjacency_0 = sparse.csr_matrix((N, N))

        if E > 0:
            rows = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
            cols = np.concatenate([np.arange(E), np.arange(E)])
            vals = np.ones(2 * E)
            self.incidence_1 = sparse.csr_matrix((vals, (rows, cols)), shape=(N, E))
        else:
            self.incidence_1 = sparse.csr_matrix((N, 0))

        if T > 0 and E > 0:
            rows, cols = [], []
            for m, (i, j, k) in enumerate(self.triangles):
                for edge in [(i, j), (i, k), (j, k)]:
                    e_idx = self.edge_to_idx.get(edge)
                    if e_idx is not None:
                        rows.append(e_idx)
                        cols.append(m)
            vals = np.ones(len(rows))
            self.incidence_2 = sparse.csr_matrix((vals, (rows, cols)), shape=(E, T))
        else:
            self.incidence_2 = sparse.csr_matrix((E, 0))

    def _build_hodge_laplacians(self):
        B1T = self.B1.T.tocsr()
        B2T = self.B2.T.tocsr()
        self.L0 = self.B1 @ B1T
        self.L1 = B1T @ self.B1 + self.B2 @ B2T
        self.L2 = B2T @ self.B2

