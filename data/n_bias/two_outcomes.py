"""edge_slope and excess against the SAME human predictor, in the same drift bands.

If they are rival measures of one construct they should track each other. If the
effects live in different drift populations they are two findings, not two
attempts at one.
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    f = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)  # noqa: E731
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

d = pd.read_parquet(REPO + "/data/sentpos_5sub.parquet")
d = d[(d.year.between(2012, 2018)) & (d.n >= 250)]
full = d.groupby("word").year.nunique()
d = d[d.word.isin(set(full[full == 7].index))].sort_values(["word", "year"])
es = d.groupby("word").edgeness.apply(lambda s: np.polyfit(np.arange(7), s.values, 1)[0])

T = pd.read_csv(REPO + "/data/step_profile/orderperm_real.csv", index_col=0,
                keep_default_na=False)
sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
dw, dv = sp.drift(2012, 2018)
J = T.join(pd.DataFrame({"edge_slope": es}), how="inner").join(conc.rename("conc"), how="inner")
J["dr"] = pd.Series(dict(zip(dw, dv))).reindex(J.index)
J = J.dropna(subset=["excess", "edge_slope", "conc", "log_freq", "dr"])
J["p"] = fm_pct(J.dr.to_numpy(), J.log_freq.to_numpy())
print(f"words with BOTH outcomes + human concreteness: {len(J):,}")
print(f"rho(edge_slope, excess) = {spearmanr(J.edge_slope, J.excess).correlation:+.4f}"
      "   <- are they even measuring the same thing?")

print("\nrho(brysbaert concreteness, OUTCOME | log_freq), frequency-matched drift deciles")
print(f"{'drift decile':<14}{'n':>7}{'edge_slope':>13}{'excess':>10}")
J["dec"] = pd.qcut(J.p, 10, labels=False, duplicates="drop")
E, X = [], []
for i in range(10):
    s = J[J.dec == i]
    e = partial(s.edge_slope, s.conc, s.log_freq); x = partial(s.excess, s.conc, s.log_freq)
    E.append(e); X.append(x)
    print(f"{i:<14}{len(s):>7,}{e:>+13.4f}{x:>+10.4f}")
print(f"\n  trend vs decile:  edge_slope {spearmanr(range(10), E).correlation:+.3f}"
      f"   excess {spearmanr(range(10), X).correlation:+.3f}")
print(f"  low-drift (0-2) mean:   edge_slope {np.mean(E[:3]):+.4f}   excess {np.mean(X[:3]):+.4f}")
print(f"  high-drift (7-9) mean:  edge_slope {np.mean(E[7:]):+.4f}   excess {np.mean(X[7:]):+.4f}")
print("\n  NOTE sign convention: edge_slope negative = concrete words peripheralize LESS")
print("  (matches the entrenchment prediction). excess positive = concrete words")
print("  travel MORE directedly (opposite the entrenchment prediction).")

# ---- bootstrap CIs on the pooled regime estimates --------------------------
print("\n" + "=" * 72)
print("BOOTSTRAP 95% CIs (2000 resamples) on the pooled regime estimates")
print("=" * 72)
rng = np.random.default_rng(11)
for lab, sel in (("low drift (deciles 0-2)", J.dec <= 2), ("high drift (deciles 7-9)", J.dec >= 7)):
    S = J[sel]
    out = {}
    for col in ("edge_slope", "excess"):
        pt = partial(S[col], S.conc, S.log_freq)
        bs = np.empty(2000)
        idx = np.arange(len(S))
        for b in range(2000):
            t = S.iloc[rng.choice(idx, len(S), replace=True)]
            bs[b] = partial(t[col], t.conc, t.log_freq)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[col] = (pt, lo, hi)
    print(f"{lab}  n={len(S):,}")
    for col, (pt, lo, hi) in out.items():
        star = "" if lo <= 0 <= hi else "  *"
        print(f"   {col:<12}{pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]{star}")
print("   * = CI excludes zero")
