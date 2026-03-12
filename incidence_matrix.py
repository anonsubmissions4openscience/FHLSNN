import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import os
import matplotlib.pyplot as plt

def get_faces(G):
    """
    Returns a list of the faces in an undirected graph
    """
    edges = list(G.edges)
    faces = []
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            if e1[0] == e2[0]:
                shared = e1[0]
                e3 = (e1[1], e2[1])
            elif e1[1] == e2[0]:
                shared = e1[1]
                e3 = (e1[0], e2[1])
            elif e1[0] == e2[1]:
                shared = e1[0]
                e3 = (e1[1], e2[0])
            elif e1[1] == e2[1]:
                shared = e1[1]
                e3 = (e1[0], e2[0])
            else: # edges don't connect
                continue

            if e3[0] in G[e3[1]]: # if 3rd edge is in graph
                faces.append(tuple(sorted((shared, *e3))))
    return list(sorted(set(faces)))


def incidence_matrices(G, V, E, faces, edge_to_idx):
    """
    Returns incidence matrices B1 and B2

    :param G: NetworkX DiGraph
    :param V: list of nodes
    :param E: list of edges
    :param faces: list of faces in G

    Returns B1 (|V| x |E|) and B2 (|E| x |faces|)
    B1[i][j]: -1 if node is is tail of edge j, 1 if node is head of edge j, else 0 (tail -> head) (smaller -> larger)
    B2[i][j]: 1 if edge i appears sorted in face j, -1 if edge i appears reversed in face j, else 0; given faces with sorted node order
    """
    B1 = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense())
    B2 = np.zeros([len(E),len(faces)])

    for f_idx, face in enumerate(faces): # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2



"""
A = np.array([
         [0, 1, 1, 1],
         [1, 0, 1, 0],
         [1, 1, 0, 1],
         [1, 0, 1, 0]])

G = nx.from_numpy_matrix(A)

edge_to_idx = {edge: i for i, edge in enumerate(G.edges)} # G.edges: [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]

B1, B2 = incidence_matrices(G, sorted(G.nodes), sorted(G.edges), get_faces(G), edge_to_idx)

L1_lower = B1.T @ B1
L1_upper = B2 @ B2.T

# scone: shifts = [L1_lower, L1_upper]
# ebli: shifts = [L1_lower + L1_upper, (L1_lower + L1_upper) @ (L1_lower + L1_upper)]

# Hodge Laplacians
L_1_hodge = L1_lower + L1_upper # shape of Hodge Laplacian is |E| * |E|

# The r-th power of the Hodge p-Laplacian: L_p^r = L_p_{down}^r + L_p_{up}^r - this equation existed!
#print(np.linalg.matrix_power(L1_lower, 2))
#print(np.linalg.matrix_power(L1_upper, 2))
#print(np.linalg.matrix_power(L_1_hodge, 2))
"""
