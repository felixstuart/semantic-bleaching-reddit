"""Does the excess content leg hold with a NON-embedding predictor?

The whole reason edge_slope outlived the 13 dead geometry metrics is that its
predictor is embedding-derived and its outcome is a raw corpus count - no shared
computational pathway. `excess` does not have that: both legs live in the same
trained space, and pred_conc is itself a ridge imputation FROM context vectors.

Human Brysbaert concreteness is external to the embedding entirely. If the leg
survives with it, excess gains the independence property. Coverage is the cost
(Brysbaert misses ~44% of the vocabulary).
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1])


def fm_pct(v, lf, dex=0.25):
    o = np.argsort(lf); l2, v2 = lf[o], v[o]
    lo = np.searchsorted(l2, l2 - dex, "left"); hi = np.searchsorted(l2, l2 + dex, "right")
    fm = np.array([(v2[lo[i]:hi[i]] < v2[i]).mean() for i in range(len(v2))])
    r = np.empty(len(v2)); r[o] = fm
    return r


brys = pd.read_csv(REPO + "/data/brysbaert_concreteness.txt", sep="\t")
brys["Word"] = brys.Word.astype(str).str.lower()
conc = brys[~brys.Word.str.contains(" ")].set_index("Word")["Conc.M"]
pred = pd.read_csv(REPO + "/data/tn_validity/predicted_concreteness.csv", index_col=0,
                   keep_default_na=False).pred_conc

sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
dw, dv = sp.drift(2012, 2018)
dr = pd.Series(dict(zip(dw, dv)))

print(f"{'arm':<10}{'predictor':<12}{'n':>8}{'all':>10}{'top40%':>10}{'top20%':>10}{'top10%':>10}")
for arm, path in (("real", "orderperm_real.csv"), ("shuffled", "orderperm_shuffled.csv")):
    T = pd.read_csv(REPO + "/data/step_profile/" + path, index_col=0, keep_default_na=False)
    T["dr"] = dr.reindex(T.index)
    for pname, P in (("brysbaert", conc), ("pred_conc", pred)):
        X = T.join(P.rename("p"), how="inner").dropna(subset=["excess", "p", "log_freq", "dr"])
        q = fm_pct(X.dr.to_numpy(), X.log_freq.to_numpy())
        row = [partial(X[q >= b].excess, X[q >= b].p, X[q >= b].log_freq)
               for b in (0.0, 0.6, 0.8, 0.9)]
        print(f"{arm:<10}{pname:<12}{len(X):>8,}" + "".join(f"{v:>+10.4f}" for v in row))

# same test against edge_slope, for a like-for-like reference point
d = pd.read_parquet(REPO + "/data/sentpos_5sub.parquet")
d = d[(d.year.between(2012, 2018)) & (d.n >= 250)]
full = d.groupby("word").year.nunique()
d = d[d.word.isin(set(full[full == 7].index))].sort_values(["word", "year"])
es = d.groupby("word").edgeness.apply(lambda s: np.polyfit(np.arange(7), s.values, 1)[0])
T = pd.read_csv(REPO + "/data/step_profile/orderperm_real.csv", index_col=0,
                keep_default_na=False)
E = pd.DataFrame({"edge_slope": es}).join(T.log_freq, how="inner").join(conc.rename("p"),
                                                                       how="inner").dropna()
print(f"\nreference: rho(brysbaert, edge_slope | freq) = "
      f"{partial(E.edge_slope, E.p, E.log_freq):+.4f}  n={len(E):,}")
