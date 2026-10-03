"""Word-centered peripheral-graph sense clustering (Kolli et al. 2026, distributional half).

Per (word, year): take the word's cached top-k neighbours, build the neighbour-to-
neighbour similarity matrix with the CENTRE EXCLUDED (the centre connects to
everything by construction, so leaving it in collapses the graph to one blob),
average-linkage cluster it, and read off two numbers -- how many sense clusters,
and what fraction of the neighbourhood sits in the largest one.

Derived per-word features are DIFFERENCES and TRENDS, never levels: levels carry
frequency (dommass_mean is -0.333 against log_freq), differences cancel it because
the frequency-driven component is roughly constant within a word. Same reason
chain_rise survives the frequency gate and chain_first does not.

scipy linkage+fcluster rather than sklearn AgglomerativeClustering: identical
result for average linkage on a precomputed distance matrix, but ~20x less
per-call overhead, which matters at ~520k clusterings. Computing the linkage tree
ONCE and cutting it at several thresholds also makes the sensitivity sweep nearly
free.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

THRESHOLDS = (0.55, 0.62, 0.70)   # similarity; distance cut = 1 - t
BLOCK = 2048


def _year_features(sp, year, thresholds, k):
    """(n_words,) arrays of n_clusters and dominant_mass for every word in `year`."""
    M = np.ascontiguousarray(sp.mat[year])
    nbr = sp.knn_idx[year][:, :k]
    n = M.shape[0]
    ncl = {t: np.empty(n, dtype=np.int16) for t in thresholds}
    dom = {t: np.empty(n, dtype=np.float32) for t in thresholds}

    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        V = M[nbr[lo:hi]]                              # (B, k, dim)
        S = np.einsum("bkd,bjd->bkj", V, V)            # (B, k, k) peripheral graph
        D = np.clip(1.0 - S, 0.0, None)
        for b in range(hi - lo):
            d = D[b]
            np.fill_diagonal(d, 0.0)
            Z = linkage(squareform(d, checks=False), method="average")
            for t in thresholds:
                lab = fcluster(Z, 1.0 - t, criterion="distance")
                counts = np.bincount(lab)
                ncl[t][lo + b] = counts.size - 1        # label 0 unused
                dom[t][lo + b] = counts.max() / float(k)
    return ncl, dom


def sense_features(sp, years, thresholds=THRESHOLDS, k=25, verbose=True):
    """Per-word dommass_change / dommass_slope / nclust_change, one frame per threshold.

    Restricted to words present in EVERY year, so the differences are well defined.
    """
    years = list(years)
    per_year = {}
    for y in years:
        t0 = time.time()
        ncl, dom = _year_features(sp, y, thresholds, k)
        per_year[y] = (ncl, dom, sp.words[y])
        if verbose:
            print(f"  [sense] {y}: {len(sp.words[y]):,} words in {time.time() - t0:.1f}s",
                  flush=True)

    common = set(per_year[years[0]][2])
    for y in years[1:]:
        common &= set(per_year[y][2])
    common = sorted(common)
    if verbose:
        print(f"  [sense] {len(common):,} words present in all {len(years)} years", flush=True)

    idx = {y: {w: i for i, w in enumerate(per_year[y][2])} for y in years}
    xs = np.asarray(years, dtype=float)
    xc = xs - xs.mean()
    denom = float((xc ** 2).sum())

    out = {}
    for t in thresholds:
        dm = np.empty((len(common), len(years)), dtype=np.float32)
        nc = np.empty((len(common), len(years)), dtype=np.float32)
        for j, y in enumerate(years):
            ncl, dom, _ = per_year[y]
            rows = np.fromiter((idx[y][w] for w in common), dtype=np.int64, count=len(common))
            dm[:, j] = dom[t][rows]
            nc[:, j] = ncl[t][rows]
        frame = pd.DataFrame(index=pd.Index(common, name="word"))
        frame["dommass_change"] = dm[:, -1] - dm[:, 0]
        frame["dommass_slope"] = (dm - dm.mean(axis=1, keepdims=True)) @ xc / denom
        frame["nclust_change"] = nc[:, -1] - nc[:, 0]
        frame["dommass_mean"] = dm.mean(axis=1)      # level: expected to be confounded
        frame["nclust_mean"] = nc.mean(axis=1)       # level: expected to be confounded
        out[t] = frame
    return out, per_year


def cluster_members(sp, word, year, threshold=0.62, k=25):
    """The actual clusters for one (word, year), for reading case studies."""
    i = sp.index[year][word]
    rows = sp.knn_idx[year][i, :k]
    V = np.asarray(sp.mat[year][rows])
    d = np.clip(1.0 - V @ V.T, 0.0, None)
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    lab = fcluster(Z, 1.0 - threshold, criterion="distance")
    ws = sp.words[year]
    groups = {}
    for j, l in zip(rows, lab):
        groups.setdefault(int(l), []).append(ws[j])
    return sorted(groups.values(), key=len, reverse=True)
