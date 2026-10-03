"""edge_slope split-half reliability, raw vs detrended.

The 0.767 figure was measured on raw edgeness. If part of it rides on a
corpus-wide sentence-length drift common to all words, detrending should
lower it. Detrending by year (subtracting the token-weighted corpus mean of
edgeness in that year) removes the shared component, of which length drift is
one part.

LIMIT, on record: this is NOT true length residualisation. sentpos_5sub.parquet
carries no sentence-length column, so per-word length cannot be regressed out.
Under uniform placement E[edgeness] falls with sentence length (0.300 at L=5,
0.278 at L=10, 0.263 at L=20), so a per-word shift in typical sentence length
moves edgeness mechanically. Testing that needs sentpos re-run on the mini with
a mean-length column. Year-detrending catches only the corpus-common part.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
YEARS = list(range(2012, 2019))
EVEN, ODD = [2012, 2014, 2016, 2018], [2013, 2015, 2017]


def slope(s):
    return np.polyfit(np.arange(len(s)), s.values, 1)[0]


def sb(r, k=2.0):
    return k * r / (1 + (k - 1) * r)


d = pd.read_parquet(REPO + "/data/sentpos_5sub.parquet")
d = d[d.year.isin(YEARS)]
print(f"{'floor':>7}{'n':>8}   {'RAW half-half':>14}{'SB':>9}   {'DETRENDED half-half':>20}{'SB':>9}")
for floor in (100, 250, 1000, 5000):
    w = d[d.n >= floor]
    full = w.groupby("word").year.nunique()
    w = w[w.word.isin(set(full[full == 7].index))].sort_values(["word", "year"])
    if len(w) == 0:
        continue
    gl = w.groupby("year").apply(lambda x: np.average(x.edgeness, weights=x.n),
                                 include_groups=False)
    w = w.assign(e=w.edgeness - w.year.map(gl))
    out = []
    for col in ("edgeness", "e"):
        a = w[w.year.isin(EVEN)].groupby("word")[col].apply(slope)
        b = w[w.year.isin(ODD)].groupby("word")[col].apply(slope)
        out.append(spearmanr(a, b).correlation)
    n = w.word.nunique()
    print(f"{floor:>7}{n:>8,}   {out[0]:>+14.4f}{sb(out[0]):>+9.4f}   "
          f"{out[1]:>+20.4f}{sb(out[1]):>+9.4f}")

# how much of edgeness is the shared year component?
w = d[d.n >= 250]
full = w.groupby("word").year.nunique()
w = w[w.word.isin(set(full[full == 7].index))]
gl = w.groupby("year").apply(lambda x: np.average(x.edgeness, weights=x.n), include_groups=False)
print(f"\ncorpus mean edgeness by year: " + " ".join(f"{v:.5f}" for v in gl.values))
b = np.polyfit(np.arange(7), gl.values, 1)[0]
print(f"  global slope {b:+.6f}/yr; over 6 years that is "
      f"{abs(b * 6) / w.edgeness.std() * 100:.1f}% of one word-level SD")
