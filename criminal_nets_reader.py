import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
import torch
import torch_geometric.datasets
from torch_geometric.data import Data
import torch_geometric.transforms as T
import os

path = "C:/Users/jaide/Python Scripts/GNN/crime_net"

meetings_edgelist = pd.read_csv(path + '/criminal_nets/Montagna_meetings_edgelist.csv').values
phonecalls_edgelist = pd.read_csv(path + '/criminal_nets/Montagna_phonecalls_edgelist.csv').values

def get_criminal_net_edge_index(input_edgelist):
    criminal_edge_index = [[i, j] for i, j in input_edgelist[:, :2]]

    clear_edge_index = []  # len of clear_edge_index is |E|*2
    for i in criminal_edge_index:
        u, v = i
        if (i in criminal_edge_index and [v, u] in criminal_edge_index):
            clear_edge_index.append(i)
            clear_edge_index.append([v, u])
            criminal_edge_index.remove([v, u])
        else:
            clear_edge_index.append(i)
            clear_edge_index.append([v, u])

    # convert edge_index to LongTensor format
    edge_index = torch.LongTensor(clear_edge_index).transpose(0, 1)

    return edge_index

def get_criminal_net_node_features(data_type, data):
    g = nx.Graph()
    if data_type == "meetings_data":
        g.add_nodes_from([i for i in range(len(data.label))])
    else:
        g.add_nodes_from([i for i in data.label])

    edge_index_ = np.array((data.edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                        range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)
    # degree, closeness, betweenness, pagerank
    degree = nx.algorithms.degree_centrality(g)
    closeness = nx.algorithms.closeness_centrality(g)
    betweenness = nx.algorithms.betweenness_centrality(g)
    pagerank = nx.pagerank(g)
    centrality_features = torch.tensor(
        [[degree[i], closeness[i], betweenness[i], pagerank[i]] for i in data.label])

    return centrality_features

def get_phonecalls_net_edges_index(data):
    g = nx.Graph()

    g.add_nodes_from([i for i in data.label])
    edge_index_ = np.array((data.edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)
    nodes_labels = list(g.nodes())
    mapping = {i: nodes_labels.index(i) for i in nodes_labels}
    # https://networkx.org/documentation/stable/reference/generated/networkx.relabel.relabel_nodes.html#networkx.relabel.relabel_nodes
    # relabel nodes
    g_ = nx.relabel_nodes(g, mapping)

    criminal_edge_index = [[i, j] for i, j in g_.edges()]

    clear_edge_index = []  # len of clear_edge_index is |E|*2
    for i in criminal_edge_index:
        u, v = i
        clear_edge_index.append(i)
        clear_edge_index.append([v,u])

    # convert edge_index to LongTensor format
    edge_index = torch.LongTensor(clear_edge_index).transpose(0, 1)

    return edge_index

# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
# for criminal networks, we set feature matrix to be trainable embedding dictionaries, i.e., \phi in the model
meetings_data_ = Data(edge_index= get_criminal_net_edge_index(meetings_edgelist), label = np.unique(meetings_edgelist[:, :2]))
phonecalls_data_ = Data(edge_index= get_criminal_net_edge_index(phonecalls_edgelist), label = np.unique(phonecalls_edgelist[:, :2]))

meetings_data = Data(x = get_criminal_net_node_features(data_type= "meetings_data", data = meetings_data_),
                     edge_index= get_criminal_net_edge_index(meetings_edgelist), y = np.unique(meetings_edgelist[:, :2]), num_classes = None)
phonecalls_data = Data(x = get_criminal_net_node_features(data_type= "phonecalls_data", data = phonecalls_data_),
                       edge_index= get_phonecalls_net_edges_index(phonecalls_data_), y = np.unique(phonecalls_edgelist[:, :2]), num_classes = None)

def get_criminal_net_edges_split(data, val_prop = 0.2, test_prop = 0.2):
    g = nx.Graph()
    phonecalls_data_ = Data(edge_index=get_criminal_net_edge_index(phonecalls_edgelist),
                            label=np.unique(phonecalls_edgelist[:, :2]))

    g.add_nodes_from([i for i in phonecalls_data_.label])
    edge_index_ = np.array((phonecalls_data_.edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)
    nodes_labels = list(g.nodes())
    mapping = {i: nodes_labels.index(i) for i in nodes_labels}
    # https://networkx.org/documentation/stable/reference/generated/networkx.relabel.relabel_nodes.html#networkx.relabel.relabel_nodes
    # relabel nodes
    g_ = nx.relabel_nodes(g, mapping)
    adj = nx.adjacency_matrix(g_)

    return get_criminal_net_adj_split(adj,val_prop = val_prop, test_prop = test_prop)

def get_criminal_net_adj_split(adj, val_prop=0.05, test_prop=0.1):
    #np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


#a,b,c,d,e,f = get_criminal_net_edges_split(phonecalls_data)
#print(a.shape)
#print(f.shape)

"""
g = nx.Graph()

g.add_nodes_from([i for i in phonecalls_data_.label])
edge_index_ = np.array((phonecalls_data_.edge_index))
edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in range(np.shape(edge_index_)[1])]
g.add_edges_from(edge_index)
nodes_labels = list(g.nodes())
mapping = {i: nodes_labels.index(i) for i in nodes_labels}
print(mapping)
# https://networkx.org/documentation/stable/reference/generated/networkx.relabel.relabel_nodes.html#networkx.relabel.relabel_nodes
# relabel nodes
g_ = nx.relabel_nodes(g, mapping)
print(g_.nodes())
"""