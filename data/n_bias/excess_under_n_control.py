"""Does the order-permutation `excess` survive control of BOTH n channels?

There are two separate n channels, and the arms control different ones:

  (1) SHARED corpus profile - 2012 is 6.3% of all pairs, 2014 is 17.6%. Every
      word is thin in 2012. The pseudo-year null HAS this profile too, so the
      real-minus-null contrast controls it.
  (2) PER-WORD deviation from that profile - real words grow and shrink
      idiosyncratically (median TV distance from bin share 0.057, 23% have
      min/max year-share ratio < 0.5). Null words do NOT (median TV 0.005,
      0.08% below 0.5). The null arm CANNOT control this channel.

Channel (2) is controlled instead by the evenness filter already used in
step_profile.py. This runs excess under both controls at once.

PRE-REGISTERED (fixed before running):
  SURVIVES  under the strictest evenness filter (min year-share ratio >= 0.7),
            real median excess >= +0.02 AND shuffled/real ratio <= 0.50
            (the same bars orderperm.py used), AND the content leg keeps sign.
  DEAD      real median excess <= +0.005 under that filter, or ratio >= 0.90.
  else      PARTIAL.
"""
import re, sys, collections
import numpy as np, pandas as pd
from scipy.stats import rankdata

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = ("/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-"
       "Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad")
YEARS = list(range(2012, 2019))
PAT = re.compile(r"^(.*)_(\d{4})$")


def load_counts(p):
    per = collections.defaultdict(dict)
    for line in open(p):
        a = line.split()
        if len(a) != 2:
            continue
        m = PAT.match(a[0])
        if m:
            per[m.group(1)][int(m.group(2))] = int(a[1])
    return per


def profile_stats(counts):
    ws = [w for w, d in counts.items() if len(d) == 7]
    M = np.array([[counts[w][y] for y in YEARS] for w in ws], dtype=np.float64)
    share = M.sum(0) / M.sum()
    P = (M / M.sum(1, keepdims=True)) / share          # ratio to corpus bin share
    msr = P.min(1) / P.max(1)                          # 1.0 = perfectly proportional
    # edge thinness: are the SEQUENCE ENDS thinner than the interior?
    edge = (P[:, [0, 6]].mean(1) - P[:, 1:6].mean(1)) / P.mean(1)
    return pd.DataFrame({"msr": msr, "edge": edge, "tot": M.sum(1)}, index=ws)


R = profile_stats(load_counts(REPO + "/vectors/5sub/counts.words.vocab"))
N = profile_stats(load_counts(SC6 + "/nullvec/counts.words.vocab"))
TR = pd.read_csv(REPO + "/data/step_profile/orderperm_real.csv", index_col=0,
                 keep_default_na=False).join(R, how="inner")
TN = pd.read_csv(REPO + "/data/step_profile/orderperm_shuffled.csv", index_col=0,
                 keep_default_na=False).join(N, how="inner")
print(f"joined: real {len(TR):,}  shuffled {len(TN):,}")

print("\n" + "=" * 74)
print("IS `excess` PREDICTED BY THE n-PROFILE?  (freq-partialled Spearman)")
print("=" * 74)


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1])


for lab, T in (("real", TR), ("shuffled", TN)):
    print(f"  {lab:<9} excess x edge-thinness {partial(T.excess, T.edge, T.log_freq):+.4f}"
          f"   excess x evenness(msr) {partial(T.excess, T.msr, T.log_freq):+.4f}")

print("\n" + "=" * 74)
print("EXCESS UNDER THE EVENNESS FILTER  (both arms, same filter)")
print("=" * 74)
print(f"{'min year-share ratio':<24}{'n real':>9}{'real med':>11}{'n shuf':>9}"
      f"{'shuf med':>11}{'ratio':>8}")
out = {}
for bar in (0.0, 0.5, 0.7, 0.8, 0.9):
    a = TR[TR.msr >= bar]; b = TN[TN.msr >= bar]
    ma, mb = a.excess.median(), b.excess.median()
    out[bar] = (len(a), ma, len(b), mb, mb / ma if ma else np.nan)
    print(f"{'>= ' + format(bar, '.2f'):<24}{len(a):>9,}{ma:>+11.4f}{len(b):>9,}"
          f"{mb:>+11.4f}{mb/ma:>8.2f}")

print("\n" + "=" * 74)
print("CONTENT LEG UNDER THE SAME FILTER  rho(pred_conc, excess | freq)")
print("=" * 74)
P = pd.read_csv(REPO + "/data/tn_validity/predicted_concreteness.csv", index_col=0,
                keep_default_na=False)
D = pd.read_csv(REPO + "/data/step_profile/orderperm_3sub.csv", index_col=0,
                keep_default_na=False)[["drift"]] if False else None
J = TR.join(P.pred_conc).dropna(subset=["excess", "pred_conc", "log_freq"])
Jn = TN.join(P.pred_conc).dropna(subset=["excess", "pred_conc", "log_freq"])
print(f"{'filter':<12}{'n':>8}{'all':>10}{'top40%':>10}{'top20%':>10}{'top10%':>10}  (drift bands)")
import sys
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402
sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
dw, dv = sp.drift(2012, 2018)
dr = pd.Series(dict(zip(dw, dv)))


def fm_pct(v, lf, dex=0.25):
    o = np.argsort(lf); l2 = lf[o]; v2 = v[o]
    lo = np.searchsorted(l2, l2 - dex, "left"); hi = np.searchsorted(l2, l2 + dex, "right")
    fm = np.array([(v2[lo[i]:hi[i]] < v2[i]).mean() for i in range(len(v2))])
    r = np.empty(len(v2)); r[o] = fm
    return r


for lab, T in (("real", J), ("shuffled", Jn)):
    for bar in (0.0, 0.7):
        X = T[T.msr >= bar].copy()
        X["dr"] = dr.reindex(X.index)
        X = X.dropna(subset=["dr"])
        X["p"] = fm_pct(X.dr.to_numpy(), X.log_freq.to_numpy())
        row = [partial(X[X.p >= q].excess, X[X.p >= q].pred_conc, X[X.p >= q].log_freq)
               for q in (0.0, 0.6, 0.8, 0.9)]
        print(f"{lab + ' >=' + format(bar, '.1f'):<12}{len(X):>8,}" +
              "".join(f"{v:>+10.4f}" for v in row))

n7, m7, _, s7, r7 = out[0.7]
verdict = ("SURVIVES" if m7 >= 0.02 and r7 <= 0.50 else
           "DEAD" if m7 <= 0.005 or r7 >= 0.90 else "PARTIAL")
print(f"\n== PRE-REGISTERED VERDICT (msr >= 0.70): real median {m7:+.4f}, "
      f"ratio {r7:.2f} -> {verdict}")
