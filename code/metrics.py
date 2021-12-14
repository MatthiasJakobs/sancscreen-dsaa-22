import torch
from scipy.stats import rankdata
from scipy.spatial import distance
import numpy as np

def L1(a, b):
    return _l1(a-b)

def L2(a, b):
    return _l2(a-b)

def _l1(a, axis=1):
    if isinstance(a, np.ndarray):
        _sum = np.sum
        _abs = np.abs
    else:
        _sum = torch.sum
        _abs = torch.abs
    if axis is None:
        return _sum(_abs(a))
    return _sum(_abs(a), axis=axis)

def _l2(a, axis=1):
    if isinstance(a, np.ndarray):
        _sqrt = np.sqrt
        _sum = np.sum
    else:
        _sqrt = torch.sqrt
        _sum = torch.sum
    if axis is None:
        return _sqrt(_sum((a)**2))
    return _sqrt(_sum((a)**2, axis=axis))

def rank(x):
    return x.argsort().argsort().astype(np.float32)

def spearman_footrule(a, b, normalize=True):
    if len(a.shape) == 1:
        a = a.reshape(1, -1)
    if len(b.shape) == 1:
        b = b.reshape(1, -1)

    a = generate_mean_ranking(a)
    b = generate_mean_ranking(b)

    upper_bound = np.floor(0.5 * (a.shape[-1]**2))

    return np.sum(np.abs(a - b), axis=1) / upper_bound

def _average_value(n, k=0):
    return (n-1)/2 + k

def generate_mean_ranking(e):
    # if len(e.shape) == 1:
    #     e = e.reshape(1, -1)
    # mean_rank = rank(e)
    # for i, v in enumerate([-3, -2, -1, 0, 1, 2, 3]):
    #     inds_r, inds_c = np.where(e == v)
    #     if len(inds_r) == 0:
    #         continue

    #     pairs = np.vstack([inds_r, inds_c]).T
    #     for j in np.unique(inds_r):
    #         relevant_pairs = pairs[np.where(pairs[:, 0] == j)][:, 1]
    #         n = len(relevant_pairs)

    #         if n > 1:
    #             av = _average_value(n, k=np.min(mean_rank[j, relevant_pairs]))
    #             mean_rank[j, relevant_pairs] = np.mean(mean_rank[j, relevant_pairs])

    #return mean_rank
    if len(e.shape) == 2 and e.shape[0] != 1:
        return rankdata(e, axis=1)
    return rankdata(e)

def assert_size_match(e0, e1):
    assert e0.size == e1.size, 'Explanations do not match in size!'

def scosine(e0, e1):
    assert_size_match(e0, e1)
    if len(e0.shape) == 2 and e0.shape[0] != 1:
        return [distance.cosine(a.flatten(), b.flatten())/2 for a, b in zip(e0, e1)]
    return distance.cosine(e0.flatten(), e1.flatten())/2

def seuclidean(e0, e1):
    assert_size_match(e0, e1)
    if len(e0.shape) == 2 and e0.shape[0] != 1:
        return [distance.euclidean(a.flatten(), b.flatten()) for a, b in zip(e0, e1)]
    return distance.euclidean(e0.flatten(), e1.flatten())/2

def shamming(e0, e1, threshold=1e-8):
    def _hamming(e0, e1):
        n = e0.size
        pos = distance.hamming(
                e0.flatten() > threshold,
                e1.flatten() > threshold)
        neg = distance.hamming(
                e0.flatten() < -threshold,
                e1.flatten() < -threshold)
        return (pos+neg)/2
    """
        Gives the proportional match between feature importances that
        are above or below a given threshold, normalized to [0, 1].
    """
    assert_size_match(e0, e1)
    if len(e0.shape) == 2 and e0.shape[0] != 1:
        return [_hamming(a, b) for a, b in zip(e0, e1)]