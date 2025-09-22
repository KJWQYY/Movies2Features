import torch
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
import argparse
import math
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def factorization_loss(f_a, f_b, w):
    # empirical cross-correlation matrix
    if f_a.size(0) > 1:
        f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
        f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    else:
        f_a_norm = (f_a - f_a.mean()) / (f_a.std()+1e-6)
        f_b_norm = (f_b - f_b.mean()) / (f_b.std()+1e-6)
    T = f_a_norm.T
    mm = torch.mm(T, f_b_norm)
    c = mm / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = w * on_diag + (1 - w) * off_diag

    return loss
def factorization_loss_M2(f_a, f_b):
    # empirical cross-correlation matrix
    w = 0.95
    if f_a.size(0) > 1:
        f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
        f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    else:
        f_a_norm = (f_a - f_a.mean()) / (f_a.std()+1e-6)
        f_b_norm = (f_b - f_b.mean()) / (f_b.std()+1e-6)
    T = f_a_norm.T
    mm = torch.mm(T, f_b_norm)
    c = mm / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    #loss = on_diag + 0.005 * off_diag 0.956+
    loss = 0.2 * on_diag + 0.8 * off_diag
    return loss
def sim_loss(A, B):

    # A_flat = A.flatten()
    # B_flat = B.flatten()

    A_normalized = F.normalize(A, p=2, dim=1)
    B_normalized = F.normalize(B, p=2, dim=1)

    cosine_similarity = F.cosine_similarity(A_normalized, B_normalized, dim=1)
    scaled_similarity = 0.5 * (cosine_similarity + 1)
    mean_cosine_similarity = torch.mean(1 - scaled_similarity)
    return mean_cosine_similarity
def topic_sim_loss(A, B, W):

    # A_flat = A.flatten()
    # B_flat = B.flatten()
    num_t = B.shape[1]
    A_expanded = A.unsqueeze(1).expand(-1, num_t, -1)


    A_normalized = F.normalize(A_expanded, p=2, dim=2)
    B_normalized = F.normalize(B, p=2, dim=2)
    #Σw*loss(A,B)
    cosine_similarity = F.cosine_similarity(A_normalized, B_normalized, dim=-1)

    dis = 0.5 * (cosine_similarity + 1)
    dis = 1-dis
    dis_w = torch.mul(dis, W)


    return torch.mean(torch.sum(dis_w, dim=1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def generate_positional_encoding( dim, max_len):
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)

    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    return pe
