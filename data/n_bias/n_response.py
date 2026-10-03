"""n-response of the neighbourhood-geometry metrics, content held constant.

The proposed decisive test was: hold content fixed, vary n, plot the metric
against n. Subsampling one year and retraining is not runnable here (the pairs
corpus is on RICK and this needs a retrain per n), but the pseudo-year null
space already IS that experiment, done once and for free:

    each word's 7 pseudo-year vectors are 7 independent estimates of the SAME
    underlying distribution, at 7 DIFFERENT token counts.

Content is constant across those 7 by construction, so any within-word
dependence of a metric on that year's n is pure estimation artifact. Using the
within-word contrast (each word centred on its own mean) removes word identity,
which the usual cross-word n-binning confounds.

Metrics tested, all from the null space:
    dev     1 - cos(v_y, own centroid)        direct noise magnitude at n
    radius  mean cos distance to k=25 NN      (= TN)
    step    1 - cos(v_y, v_y+1)               drift per transition
    chain   max cos(v_y+1, nbrs(w,y))         chaining / detachment

PRE-REGISTERED VERDICTS (fixed BEFORE running):
  N-BIASED    within-word median Spearman(metric, n) <= -0.30 (or >= +0.30)
              AND same sign in >=8/10 frequency deciles.
  CLEAN       |within-word median Spearman| <= 0.10.
  else        PARTIAL.
An unbiased-in-expectation metric (edgeness is the reference case) must land
CLEAN; anything landing N-BIASED cannot carry a temporal trend on its own.
"""
import re, sys, collections, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = ("/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-"
       "Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad")
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402

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


def within_word_rho(M, Nn):
    """Row-wise Spearman between metric M (n_words,7) and counts Nn (n_words,7)."""
    rm = rankdata(M, axis=1); rn = rankdata(Nn, axis=1)
    rm = rm - rm.mean(1, keepdims=True); rn = rn - rn.mean(1, keepdims=True)
    num = (rm * rn).sum(1)
    den = np.sqrt((rm ** 2).sum(1) * (rn ** 2).sum(1)) + 1e-12
    return num / den


NC = load_counts(SC6 + "/nullvec/counts.words.vocab")
sp = Space(SC6 + "/nullvec/null_iter15.words", cache_dir=SC6 + "/cache_null", verbose=False)
words = [w for w in sp.words[2012]
         if all(w in sp.index[y] for y in YEARS) and len(NC.get(w, {})) == 7]
print(f"null space: {len(words):,} words present in all 7 pseudo-years", flush=True)

Nn = np.array([[NC[w][y] for y in YEARS] for w in words], dtype=np.float64)
logf = np.log10(Nn.sum(1))

# ---- dev: distance from the word's own centroid, computed in chunks --------
dev = np.empty((len(words), 7), dtype=np.float64)
rows = {y: np.array([sp.index[y][w] for w in words], dtype=np.int64) for y in YEARS}
CH = 4096
for lo in range(0, len(words), CH):
    hi = min(lo + CH, len(words))
    V = np.stack([np.asarray(sp.mat[y][rows[y][lo:hi]]) for y in YEARS], axis=1)
    c = V.mean(1); c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
    dev[lo:hi] = 1.0 - np.einsum("nyd,nd->ny", V, c)
print("dev computed", flush=True)

# ---- radius / step / chain from the cached chaining table -----------------
ch = sp.chaining()
ch = ch[ch.word.isin(set(words))]
MET = {"dev": (dev, Nn)}
pos = {w: i for i, w in enumerate(words)}
TY = list(range(2012, 2018))          # 6 transition years
Nt = Nn[:, :6]
Nh = 2.0 / (1.0 / Nn[:, :6] + 1.0 / Nn[:, 1:])   # harmonic mean of the two endpoints
for name in ("radius", "step", "chain"):
    A = np.full((len(words), 6), np.nan)
    sub = ch[["word", "year", name]].dropna()
    sub = sub[sub.year.isin(TY)]
    A[[pos[w] for w in sub.word], [y - 2012 for y in sub.year]] = sub[name].to_numpy()
    MET[name] = (A, Nh if name == "step" else Nt)

print("\n" + "=" * 78)
print("WITHIN-WORD n-RESPONSE  (null space: content constant across the 7 bins)")
print("=" * 78)
print(f"{'metric':<8}{'n words':>9}{'median rho':>12}{'% neg':>8}{'deciles same sign':>20}")
res = {}
for name, (A, NX) in MET.items():
    ok = ~np.isnan(A).any(1)
    r = within_word_rho(A[ok], NX[ok])
    med = float(np.median(r))
    dec = pd.qcut(logf[ok], 10, labels=False, duplicates="drop")
    dm = [float(np.median(r[dec == i])) for i in range(10)]
    same = sum((d < 0) == (med < 0) for d in dm)
    res[name] = dict(median_rho=med, pct_neg=float((r < 0).mean()), decile_medians=dm,
                     n=int(ok.sum()))
    print(f"{name:<8}{ok.sum():>9,}{med:>+12.4f}{(r<0).mean()*100:>7.1f}%{same:>17}/10")
    print(f"{'':8}by freq decile (rare->frequent): " + " ".join(f"{d:+.2f}" for d in dm))

print("\nVERDICTS (pre-registered bars)")
for name, d in res.items():
    m = d["median_rho"]
    same = sum((x < 0) == (m < 0) for x in d["decile_medians"])
    v = "N-BIASED" if abs(m) >= 0.30 and same >= 8 else ("CLEAN" if abs(m) <= 0.10 else "PARTIAL")
    print(f"  {name:<8} rho {m:+.4f}  ->  {v}")

# ---- the requested figure: metric vs n, word identity differenced out ------
print("\n" + "=" * 78)
print("METRIC vs n, WITHIN-WORD CENTRED (each word's own mean subtracted)")
print("raw pair counts; chaining.py notes ~8 pairs per raw token")
print("=" * 78)
edges = np.array([1.2e3, 5e3, 2e4, 8e4, 3e5, 1.2e6, 5e6, 1e12])
lab = ["1.2k-5k", "5k-20k", "20k-80k", "80k-300k", "300k-1.2M", "1.2M-5M", ">5M"]
tab = {}
for name, (A, NX) in MET.items():
    ok = ~np.isnan(A).any(1)
    C = A[ok] - A[ok].mean(1, keepdims=True)
    b = np.digitize(NX[ok], edges) - 1
    tab[name] = [float(np.median(C[b == i])) if (b == i).sum() > 200 else np.nan
                 for i in range(7)]
T = pd.DataFrame(tab, index=lab)
T.index.name = "pairs in bin"
print(T.round(5).to_string())
print("\nmonotone in n?  " + "  ".join(
    f"{c}={'YES' if np.all(np.diff(T[c].dropna().to_numpy()) < 0) or np.all(np.diff(T[c].dropna().to_numpy()) > 0) else 'no'}"
    for c in T.columns))
T.to_csv(REPO + "/data/n_bias/n_response_curve.csv")
json.dump(res, open(REPO + "/data/n_bias/n_response.json", "w"), indent=1)
