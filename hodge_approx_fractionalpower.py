
import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
from scipy.linalg import fractional_matrix_power


def chebyshev_coefficients(alpha, K, N_quad=200):
    nodes = np.cos(np.pi * (np.arange(N_quad) + 0.5) / N_quad)  
    g_vals = ((nodes + 1.0) / 2.0) ** alpha

    coeffs = np.zeros(K + 1)
    for j in range(K + 1):
        T_j = np.cos(j * np.arccos(nodes))
        coeffs[j] = (2.0 / N_quad) * np.dot(g_vals, T_j)

    return coeffs


def estimate_lambda_max(L, method='eigsh'):

    if method == 'eigsh':
        L_sp = csr_matrix(L) if not issparse(L) else L
        return float(eigsh(L_sp, k=1, which='LM', return_eigenvectors=False)[0])
    elif method == 'bound':
        if issparse(L):
            row_sums = np.array(np.abs(L).sum(axis=1)).flatten()
        else:
            row_sums = np.abs(L).sum(axis=1)
        return float(np.max(row_sums))


def fractional_chebyshev(L, x, alpha, K=50, lam_max=None):
    n = L.shape[0]

    if lam_max is None:
        lam_max = estimate_lambda_max(L)
    if lam_max < 1e-12:
        return np.zeros_like(x)

    coeffs = chebyshev_coefficients(alpha, K)
    L_tilde = (2.0 / lam_max) * L - np.eye(n)
    v_prev = x.copy()                     
    v_curr = L_tilde @ x                    
    y = (coeffs[0] / 2.0) * v_prev + coeffs[1] * v_curr
    for j in range(2, K + 1):
        v_next = 2.0 * (L_tilde @ v_curr) - v_prev  
        y += coeffs[j] * v_next
        v_prev = v_curr
        v_curr = v_next
    y *= lam_max ** alpha

    return y

def get_faces(G):
    """Extract 2-simplices (triangles) from graph."""
    edges = list(G.edges)
    faces = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1, e2 = edges[i], edges[j]
            if e1[0] == e2[0]:
                shared, e3 = e1[0], (e1[1], e2[1])
            elif e1[1] == e2[0]:
                shared, e3 = e1[1], (e1[0], e2[1])
            elif e1[0] == e2[1]:
                shared, e3 = e1[0], (e1[1], e2[0])
            elif e1[1] == e2[1]:
                shared, e3 = e1[1], (e1[0], e2[0])
            else:
                continue
            if e3[0] in G[e3[1]]:
                faces.append(tuple(sorted((shared, *e3))))
    return list(sorted(set(faces)))



def compute_hodge_laplacian(B1, B2):
    return B1.T @ B1 + B2 @ B2.T


def fractional_exact(L, x, alpha):
    L_alpha = fractional_matrix_power(L, alpha)
    if x is not None:
        return L_alpha @ x
    return L_alpha



def incidence_matrices(G, V, E, faces, edge_to_idx):
    """Build B1 (node-edge) and B2 (edge-triangle) incidence matrices."""
    import networkx as nx
    B1 = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense())
    B2 = np.zeros([len(E), len(faces)])
    for f_idx, face in enumerate(faces):
        edges_f = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges_f]
        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2


#----> test shur pade vs cheb approx. 


G = nx.Graph()
G.add_edges_from([
    (0, 1), (1, 2), (2, 0),
    (2, 3), (3, 4), (4, 2),
    (0, 4), (1, 3), (3, 5),
    (5, 4), (5, 1),
])
V = sorted(G.nodes)
E = sorted(G.edges)
edge_to_idx = {e: i for i, e in enumerate(E)}
faces = get_faces(G)
B1, B2 = incidence_matrices(G, V, E, faces, edge_to_idx)
L1 = compute_hodge_laplacian(B1, B2)
print(L1)
n_edges = L1.shape[0]
x = np.random.randn(n_edges)

for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
    y_exact = fractional_exact(L1, x, alpha)
    y_cheb = fractional_chebyshev(L1, x, alpha, K=50)
    rel = np.linalg.norm(y_exact - y_cheb) / (np.linalg.norm(y_exact) + 1e-12)
    print(f"  alpha={alpha:.2f} | rel error: {rel:.2e}")


y_exact = fractional_exact(L1, x, 0.5)
for K in [5, 10, 20, 30, 50, 80]:
    y_cheb = fractional_chebyshev(L1, x, 0.5, K=K)
    rel = np.linalg.norm(y_exact - y_cheb) / (np.linalg.norm(y_exact) + 1e-12)
    print(f"  K={K:3d} | rel error: {rel:.2e}")
