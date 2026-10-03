"""edge_slope Spearmans, redone floor-graded, plus the band audit applied to it.

edge_slope is a corpus statistic with no embedding in it, so it has no
null-space twin. But the DRIFT BANDING is embedding-derived, so banding real
edge_slope by NULL-space drift tests whether the band structure alone
manufactures a gradient - the same test that cleared excess.
"""
import re, os, sys, collections
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = ("/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-"
       "Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad")
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402
YEARS = list(range(2012, 2019))


def partial(x, y, z):
    m = ~(pd.isna(x) | pd.isna(y) | pd.isna(z))
    rx, ry, rz = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1]), int(m.sum())


def fm_pct(v, lf, dex=0.25):
    o = np.argsort(lf); l2, v2 = lf[o], v[o]
    lo = np.searchsorted(l2, l2 - dex, "left"); hi = np.searchsorted(l2, l2 + dex, "right")
    fm = np.array([(v2[lo[i]:hi[i]] < v2[i]).mean() for i in range(len(v2))])
    r = np.empty(len(v2)); r[o] = fm
    return r


brys = pd.read_csv(REPO + "/data/brysbaert_concreteness.txt", sep="\t")
brys["Word"] = brys.Word.astype(str).str.lower()
conc = brys[~brys.Word.str.contains(" ")].drop_duplicates("Word").set_index("Word")["Conc.M"]
T = pd.read_csv(REPO + "/data/step_profile/orderperm_real.csv", index_col=0,
                keep_default_na=False)
d0 = pd.read_parquet(REPO + "/data/sentpos_5sub.parquet")
d0 = d0[d0.year.isin(YEARS)]
sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
dw, dv = sp.drift(2012, 2018)
drift_real = pd.Series(dict(zip(dw, dv)))
spn = Space(SC6 + "/nullvec/null_iter15.words", cache_dir=SC6 + "/cache_null", verbose=False)
nw, nv = spn.drift(2012, 2018)
drift_null = pd.Series(dict(zip(nw, nv)))


def build(floor):
    w = d0[d0.n >= floor]
    full = w.groupby("word").year.nunique()
    w = w[w.word.isin(set(full[full == 7].index))].sort_values(["word", "year"])
    es = w.groupby("word").edgeness.apply(lambda s: np.polyfit(np.arange(7), s.values, 1)[0])
    D = pd.DataFrame({"edge_slope": es})
    D["conc"] = conc.reindex(D.index)
    D["log_freq"] = T.log_freq.reindex(D.index)
    D["excess"] = T.excess.reindex(D.index)
    D["dr"] = drift_real.reindex(D.index)
    D["dn"] = drift_null.reindex(D.index)
    return D.dropna(subset=["edge_slope", "log_freq"])


print("A. rho(brysbaert, edge_slope | log_freq) by sentpos token floor")
print(f"{'floor':>7}{'n':>8}{'raw':>10}{'freq-partialled':>18}")
for floor in (100, 250, 1000, 5000):
    D = build(floor)
    r, n_ = partial(D.edge_slope.to_numpy(), D.conc.to_numpy(), D.log_freq.to_numpy())
    raw = spearmanr(D.edge_slope, D.conc, nan_policy="omit").correlation
    print(f"{floor:>7}{n_:>8,}{raw:>+10.4f}{r:>+18.4f}")

print("\nB. rho(edge_slope, excess) - do the two outcomes measure the same thing?")
print(f"{'floor':>7}{'n':>8}{'raw':>10}{'freq-partialled':>18}")
for floor in (100, 250, 1000, 5000):
    D = build(floor).dropna(subset=["excess"])
    r, n_ = partial(D.edge_slope.to_numpy(), D.excess.to_numpy(), D.log_freq.to_numpy())
    print(f"{floor:>7}{n_:>8,}{spearmanr(D.edge_slope, D.excess).correlation:>+10.4f}{r:>+18.4f}")

print("\nC. BAND AUDIT applied to edge_slope: rho(brysbaert, edge_slope | freq)")
print("   banded by REAL drift vs by NULL drift (freq-matched percentile)")
BANDS = [("all", 0.0), ("top40%", 0.6), ("top20%", 0.8), ("top10%", 0.9)]
for floor in (250, 1000):
    print(f"  floor {floor}")
    for lab, col in (("real drift", "dr"), ("NULL drift", "dn")):
        D = build(floor).dropna(subset=["conc", col])
        p = fm_pct(D[col].to_numpy(), D.log_freq.to_numpy())
        row = [partial(D[p >= b].edge_slope.to_numpy(), D[p >= b].conc.to_numpy(),
                       D[p >= b].log_freq.to_numpy())[0] for _, b in BANDS]
        print(f"    {lab:<12}n={len(D):>6,}" + "".join(f"{v:>+11.4f}" for v in row))
