
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils import set_seed


class FHLSNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32,
                 use_simplicial: bool = True, dropout: float = 0.5):
        super().__init__()
        self.use_simplicial = use_simplicial
        self.node1 = nn.Linear(in_dim, hidden)
        self.node2 = nn.Linear(hidden, out_dim)
        if use_simplicial:
            self.edge1 = nn.Linear(in_dim, hidden)
            self.edge2 = nn.Linear(hidden, out_dim)
            self.fuse = nn.Linear(2 * out_dim, out_dim)
        self.scorer = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, filt_node, filt_edge, edge_to_node, edge_feats, gamma):
        h = F.relu(self.node1(self.dropout(X)))
        h = filt_node.apply(h, gamma)
        h = self.node2(self.dropout(h))
        h = filt_node.apply(h, gamma)

        if self.use_simplicial:
            e = F.relu(self.edge1(self.dropout(edge_feats)))
            e = filt_edge.apply(e, gamma)
            e = self.edge2(e)
            e = filt_edge.apply(e, gamma)
            h = self.fuse(torch.cat([h, torch.sparse.mm(edge_to_node, e)], dim=1))
        return h

    def decode(self, z, pairs):
        u, v = z[pairs[:, 0]], z[pairs[:, 1]]
        return self.scorer(torch.cat([u * v, (u - v).abs()], dim=1)).squeeze(-1)


def train_link_prediction(X, splits, filt_node, filt_edge, edge_to_node,
                          edge_feats, gamma: float, seed: int = 0,
                          epochs: int = 120, lr: float = 0.01,
                          weight_decay: float = 5e-4, patience: int = 10,
                          eval_every: int = 5, use_simplicial: bool = True,
                          device=None, verbose: bool = False) -> dict:
    """Train and return best-validation test metrics."""
    device = device or torch.device("cpu")
    set_seed(seed)
    model = FHLSNN(X.size(1), use_simplicial=use_simplicial).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.long, device=device)
    tr_pos, tr_neg = T(splits["train_pos"]), T(splits["train_neg"][:len(splits["train_pos"])])
    va_pos, va_neg = T(splits["val_pos"]), T(splits["val_neg"])
    te_pos, te_neg = T(splits["test_pos"]), T(splits["test_neg"])

    def evaluate(pos, neg):
        model.eval()
        with torch.no_grad():
            z = model(X, filt_node, filt_edge, edge_to_node, edge_feats, gamma)
            scores = torch.cat([model.decode(z, pos), model.decode(z, neg)]).cpu().numpy()
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        return roc_auc_score(y, scores), average_precision_score(y, scores)

    best_val, best_test, bad = -1.0, (0.0, 0.0), 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z = model(X, filt_node, filt_edge, edge_to_node, edge_feats, gamma)
        logits = torch.cat([model.decode(z, tr_pos), model.decode(z, tr_neg)])
        y = torch.cat([torch.ones(len(tr_pos)), torch.zeros(len(tr_neg))]).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

        if ep % eval_every == 0 or ep == epochs - 1:
            val_auc, _ = evaluate(va_pos, va_neg)
            if val_auc > best_val:
                best_val, best_test, bad = val_auc, evaluate(te_pos, te_neg), 0
            else:
                bad += 1
                if bad > patience:
                    break
            if verbose:
                print(f"    epoch {ep:3d} loss {loss.item():.4f} val {val_auc:.4f}")

    return dict(test_auc=best_test[0], test_ap=best_test[1], val_auc=best_val)
