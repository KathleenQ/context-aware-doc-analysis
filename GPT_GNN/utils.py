from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import torch
from texttable import Texttable


def args_print(args):
    _dict = vars(args)
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum == 0] = -1
    r_inv = np.power(rowsum, -1).flatten()
    del rowsum
    r_inv[r_inv == -1] = 0.
    r_mat_inv = sp.diags(r_inv)
    del r_inv
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def randint():
    return np.random.randint(2 ** 32 - 1)


def feature_OAG(layer_data, graph):
    feature = {}
    times = {}
    indxs = {}
    attr = np.array([])
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs = np.array(list(layer_data[_type].keys()))
        tims = np.array(list(layer_data[_type].values()))[:, 1]

        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        emb = np.array(list(graph.node_feature[_type].loc[idxs, 'emb']))
        if len(emb.shape) == 1:
            emb = emb[:, None]  # I added (to concatenate matrix with matrix - the same length)

        # ADD for graph considering abstract contents:
        abs_emb = np.array(list(graph.node_feature[_type].loc[idxs, 'abs_emb']))
        if len(abs_emb.shape) == 1:
            abs_emb = abs_emb[:, None]
        feature[_type] = np.concatenate((feature[_type], emb, abs_emb, np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)

        # # NOT consider abstract contents:
        # feature[_type] = np.concatenate((feature[_type], emb, np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)

        times[_type] = tims
        indxs[_type] = idxs

        if _type == 'paper':
            attr = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, attr


def load_gnn(_dict):
    out_dict = {}
    for key in _dict:
        if 'gnn' in key:
            out_dict[key[4:]] = _dict[key]
    return OrderedDict(out_dict)
