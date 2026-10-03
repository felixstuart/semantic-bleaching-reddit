"""Participation ratio of the local neighbourhood + raw vector norm, real vs null.

PR = (sum lambda)^2 / sum(lambda^2) over the eigenvalues of the CENTRED neighbour
covariance. 1 = neighbours strung along one direction, k = isotropic.
No eigendecomposition needed: for symmetric G, sum(lambda)=tr(G) and
sum(lambda^2)=||G||_F^2, so PR = tr(G)^2 / ||G||_F^2 straight off the Gram matrix.

PREDICTION ON RECORD (before running): PR is scale-invariant, which is the exact
property that made concentration null-identical, and its scale-carrying counterpart
TN is already null-identical (0.3285 vs 0.3323). Expect PR to match the null too.
Vector norm is expected to be strongly frequency-correlated.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
REPO = Path("/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift")
sys.path.insert(0, "."); sys.path.insert(0, str(REPO))
from driftcache import Space
from scipy.stats import spearmanr

YEARS = list(range(2012, 2019))
BLOCK = 1024


def pr_year(sp, year, k):
    M = np.ascontiguousarray(sp.mat[year])
    nbr = sp.knn_idx[year][:, :k]
    n = M.shape[0]
    out = np.empty(n, dtype=np.float32)
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        V = M[nbr[lo:hi]].astype(np.float64)          # (B,k,d)
        V -= V.mean(axis=1, keepdims=True)            # centre the neighbour cloud
        G = np.einsum("bkd,bjd->bkj", V, V)           # (B,k,k) Gram, same spectrum
        tr = np.einsum("bkk->b", G)
        fro = np.einsum("bkj,bkj->b", G, G)
        out[lo:hi] = np.where(fro > 0, tr * tr / fro, np.nan)
    return out


def norms_from_raw(path, limit=None):
    """L2 norms straight from the word2vec text dump (driftcache normalises away)."""
    out = {}
    with open(path) as f:
        nrow, dim = (int(x) for x in f.readline().split())
        for i, line in enumerate(f):
            key, rest = line.split(" ", 1)
            v = np.fromstring(rest, dtype=np.float32, sep=" ")
            out[key] = float(np.sqrt((v * v).sum()))
            if limit and i + 1 >= limit:
                break
    return out


def build(vec, cache, counts_path, label, k=25):
    sp = Space(str(vec), cache_dir=str(cache), verbose=False)
    pr = {y: dict(zip(sp.words[y], pr_year(sp, y, k))) for y in YEARS}
    common = set(sp.words[YEARS[0]])
    for y in YEARS[1:]:
        common &= set(sp.words[y])
    common = sorted(common)
    M = np.array([[pr[y].get(w, np.nan) for y in YEARS] for w in common])
    xs = np.array(YEARS, float); xc = xs - xs.mean()
    df = pd.DataFrame(index=pd.Index(common, name="word"))
    df["pr_first"] = M[:, 0]; df["pr_last"] = M[:, -1]
    df["pr_mean"] = np.nanmean(M, axis=1)
    df["pr_change"] = M[:, -1] - M[:, 0]
    df["pr_slope"] = np.nansum((M - np.nanmean(M, axis=1, keepdims=True)) * xc, axis=1) / (xc ** 2).sum()
    counts = sp.counts(str(counts_path))
    df["log_freq"] = pd.Series({w: np.log10(sum(d.values())) for w, d in counts.items()})
    print(f"  {label}: {len(df):,} words", flush=True)
    return df
