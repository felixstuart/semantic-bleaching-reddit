"""Split-half reliability of PER-WORD excess, on one space.

Until this exists, no null computed with excess is interpretable: a set of
effects clustered at 0.00-0.04 is either a real absence or a noise ceiling, and
only the reliability tells you which.

Method matches how edge_slope's 0.767 was obtained - interleaved year halves of
one instrument, Spearman-Brown corrected. Two splits are reported because a
sequence statistic is sensitive to how the sequence is cut:
  4/3  [2012,2014,2016,2018] vs [2013,2015,2017]   (edge_slope's split)
  3/3  [2012,2014,2016]      vs [2013,2015,2017]   (balanced, equal length)
Permutation baselines are EXACT here (4! = 24, 3! = 6 orderings enumerated),
so no Monte-Carlo error enters the reliability estimate.
"""
import sys, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402

YEARS = list(range(2012, 2019))
sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
words = [w for w in sp.words[2012] if all(w in sp.index[y] for y in YEARS)]
rows = {y: np.array([sp.index[y][w] for w in words], dtype=np.int64) for y in YEARS}
V = np.stack([np.asarray(sp.mat[y][rows[y]]) for y in YEARS], axis=1).astype(np.float32)
print(f"{len(words):,} words present in all 7 years")


def dir_pers(V):
    D = V[:, 1:, :] - V[:, :-1, :]
    Dn = D / (np.linalg.norm(D, axis=2, keepdims=True) + 1e-12)
    return np.einsum("ntd,ntd->nt", Dn[:, 1:, :], Dn[:, :-1, :]).mean(1)


def excess_exact(sub):
    """excess with the permutation baseline enumerated exactly over all orderings."""
    W = V[:, [YEARS.index(y) for y in sub], :]
    obs = dir_pers(W)
    perms = list(itertools.permutations(range(len(sub))))
    base = np.mean([dir_pers(W[:, list(p), :]) for p in perms], axis=0)
    return obs - base


def sb(r, k):
    """Spearman-Brown: r is the half-half correlation, k the length ratio."""
    return k * r / (1 + (k - 1) * r)


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1])


T = pd.read_csv(REPO + "/data/step_profile/orderperm_real.csv", index_col=0,
                keep_default_na=False)
lf = T.log_freq.reindex(words).to_numpy()
full = T.excess.reindex(words).to_numpy()
ok = ~np.isnan(lf)

print("\n" + "=" * 70)
print("SPLIT-HALF RELIABILITY OF PER-WORD excess")
print("=" * 70)
for lab, A, B, k in (("4/3 interleaved", [2012, 2014, 2016, 2018], [2013, 2015, 2017], 2.0),
                     ("3/3 balanced", [2012, 2014, 2016], [2013, 2015, 2017], 7 / 3)):
    a, b = excess_exact(A), excess_exact(B)
    r = spearmanr(a[ok], b[ok]).correlation
    rp = partial(a[ok], b[ok], lf[ok])
    print(f"{lab:<18} half-half rho {r:+.4f}   Spearman-Brown {sb(r, k):+.4f}")
    print(f"{'':18} freq-partialled {rp:+.4f}   SB {sb(rp, k):+.4f}")
    print(f"{'':18} each half vs FULL 7-year excess: "
          f"{spearmanr(a[ok], full[ok]).correlation:+.3f} / "
          f"{spearmanr(b[ok], full[ok]).correlation:+.3f}")

# ---- floor-graded, the way edge_slope's reliability was reported ------------
print("\nby token floor (pairs), 4/3 split:")
a, b = excess_exact([2012, 2014, 2016, 2018]), excess_exact([2013, 2015, 2017])
counts = sp.counts(REPO + "/vectors/5sub/counts.words.vocab")
mn = np.array([min(counts[w].get(y, 0) for y in YEARS) if w in counts else 0 for w in words])
for floor in (0, 1200, 5000, 20000, 100000):
    m = ok & (mn >= floor)
    if m.sum() < 200:
        continue
    r = spearmanr(a[m], b[m]).correlation
    print(f"  min-year >= {floor:>7,} pairs  n={m.sum():>6,}  half-half {r:+.4f}  SB {sb(r, 2.0):+.4f}")
