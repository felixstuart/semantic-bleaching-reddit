"""chain_slope and step_slope, real vs pseudo-year null (content constant)."""
import sys
import numpy as np
REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = ("/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-"
       "Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad")
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402


def slopes(sp, label):
    ch = sp.chaining().dropna(subset=["chain", "step", "ratio"])
    keep = ch.groupby("word").year.nunique() == 6
    ch = ch[ch.word.isin(set(keep[keep].index))].sort_values(["word", "year"])
    A = ch.chain.to_numpy().reshape(-1, 6); S = ch.step.to_numpy().reshape(-1, 6)
    x = np.arange(6.) - 2.5
    cs = (A * x).sum(1) / (x ** 2).sum(); ss = (S * x).sum(1) / (x ** 2).sum()
    print(f"[{label}] n={len(A):,}")
    print(f"   chain_slope median {np.median(cs):+.5f}  %positive {(cs > 0).mean() * 100:.1f}%")
    print(f"   step_slope  median {np.median(ss):+.5f}  %negative {(ss < 0).mean() * 100:.1f}%")
    print("   mean chain by year: " + " ".join(f"{v:.4f}" for v in A.mean(0)))
    return np.median(cs), np.median(ss)


r = slopes(Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False), "REAL")
n = slopes(Space(SC6 + "/nullvec/null_iter15.words", cache_dir=SC6 + "/cache_null", verbose=False),
           "NULL (content constant)")
print(f"\nnull/real chain_slope {n[0] / r[0]:.1f}x   step_slope {n[1] / r[1]:.1f}x")
