#Finding NN of embeddings for our NNCL implementation
import os
import numpy as np
import torch
import torch.nn as nn

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def NN(key, queue, num_neighbors=1, return_indices=False):
    """
    key: N x D matrix
    queue: M x D matrix
    
    output: num_neighbors x N x D matrix for closest neighbors of key within queue
    NOTE: Output is unnormalized
    """
    #Apply Cosine similarity (equivalent to l2-normalization + dot product)
    similarity = sim_matrix(key, queue)
    
    indices_top_neighbors = torch.topk(similarity, k=num_neighbors, dim=1)[1]
    
    list_top_neighbors = []
    for i in range(num_neighbors):
        indices_ith_neighbor = indices_top_neighbors[:,i]
        list_top_neighbors.append(queue[indices_ith_neighbor,:])
        
    if return_indices:
        return torch.stack(list_top_neighbors), indices_top_neighbors
    else:
        return torch.stack(list_top_neighbors)