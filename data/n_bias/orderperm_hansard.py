#!/usr/bin/env python3
"""Order-permutation `excess` on the Hansard TR space, with a HUMAN predictor.

Cross-century, cross-register, cross-variety replication of the content ->
trajectory-type finding. Self-contained: order-permutation needs no null
retrain, and Brysbaert is external to the embedding, so the only Hansard
artifact required is the trained space + its counts vocab.

Runs on the MINI (this needs ~/hansard). Paths are flags - point them at
whatever the Hansard embed actually wrote.

    python orderperm_hansard.py \
        --vectors ~/hansard/vectors/sgns.words \
        --counts  ~/hansard/vectors/counts.words.vocab \
        --brysbaert <repo>/data/brysbaert_concreteness.txt \
        --driftcache <repo>

WHY THE MISSING SHUFFLED ARM IS SURVIVABLE HERE. On Reddit the falsification
arm existed to absorb one mechanical channel: unequal bin sizes make the thin
2012 vector noisy, and it sits at the sequence EDGE in observed order, where a
permutation moves it inward and does more anticorrelation damage. That inflates
excess with zero content. The Hansard embed was capped at 42M tokens/decade, so
bins are near-equal BY CONSTRUCTION and the channel should be small. The script
measures it (step 1) instead of assuming it - if edge-thinness is near zero and
uncorrelated with excess, the mechanical floor is not doing the work.

PRE-REGISTERED VERDICTS (fix these BEFORE looking at the output; bars carried
over from the 3-sub confirmation so this is the same standard):
  REPLICATES   top-20%-drift band rho(brysbaert, excess | freq) >= +0.08,
               monotone across the four drift bands,
               AND median excess >= +0.02 with >= 8/10 freq deciles positive.
  DEAD         top-20% band rho <= +0.02, or median excess <= +0.005.
  else         PARTIAL.
  Sign note: the Reddit result is POSITIVE (concrete -> more directed paths),
  which is opposite the entrenchment prediction in TNPosPredictor cell 0. A
  NEGATIVE Hansard rho of similar magnitude is not a replication - it is a
  contradiction, and must be reported as one.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--vectors", required=True)
ap.add_argument("--counts", required=True)
ap.add_argument("--brysbaert", required=True)
ap.add_argument("--driftcache", required=True, help="repo root, for driftcache.py")
ap.add_argument("--cache", default="./cache_hansard")
ap.add_argument("--bins", default="1920,1930,1940,1950,1960,1970,1980,1990,2000,2010")
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--out", default="orderperm_hansard.csv")
a = ap.parse_args()

sys.path.insert(0, a.driftcache)
from driftcache import Space  # noqa: E402

BINS = [int(x) for x in a.bins.split(",")]
rng = np.random.default_rng(7)


def dir_pers_of(V):
    """V: (n_words, n_bins, dim). Mean cos between consecutive displacements."""
    D = V[:, 1:, :] - V[:, :-1, :]
    Dn = D / (np.linalg.norm(D, axis=2, keepdims=True) + 1e-12)
    return np.einsum("ntd,ntd->nt", Dn[:, 1:, :], Dn[:, :-1, :]).mean(1)


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    f = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1])


def fm_pct(v, lf, dex=0.25):
    """Percentile among peers within +-dex log-frequency. Raw drift selects rare
    words, so never band on it directly."""
    o = np.argsort(lf); l2, v2 = lf[o], v[o]
    lo = np.searchsorted(l2, l2 - dex, "left"); hi = np.searchsorted(l2, l2 + dex, "right")
    fm = np.array([(v2[lo[i]:hi[i]] < v2[i]).mean() for i in range(len(v2))])
    r = np.empty(len(v2)); r[o] = fm
    return r


sp = Space(os.path.expanduser(a.vectors), cache_dir=a.cache,
           years=tuple(BINS), verbose=True)
words = [w for w in sp.words[BINS[0]] if all(w in sp.index[y] for y in BINS)]
print(f"\n{len(words):,} words present in all {len(BINS)} decade bins", flush=True)
if len(words) < 2000:
    sys.exit("too few words in every bin - check --bins against the vector key suffixes")

counts = sp.counts(os.path.expanduser(a.counts))
C = np.array([[counts.get(w, {}).get(y, 0) for y in BINS] for w in words], dtype=np.float64)
keep = (C > 0).all(1)
words = [w for w, k in zip(words, keep) if k]
C = C[keep]
print(f"{len(words):,} with a nonzero count in every bin")

# ---- 1. how uneven are the bins? (the channel the shuffled arm absorbed) ----
share = C.sum(0) / C.sum()
print("\nper-bin share of all pairs: " + " ".join(f"{s:.3f}" for s in share))
print(f"  max/min bin share = {share.max() / share.min():.2f}   "
      f"(Reddit was 17.6/6.3 = 2.79)")
P = (C / C.sum(1, keepdims=True)) / share
msr = P.min(1) / P.max(1)
edge = (P[:, [0, -1]].mean(1) - P[:, 1:-1].mean(1)) / P.mean(1)
print(f"  per-word min/max year-share ratio: median {np.median(msr):.3f}")
print(f"  edge-thinness: median {np.median(edge):+.4f}")

# ---- 2. excess -------------------------------------------------------------
rows = {y: np.array([sp.index[y][w] for w in words], dtype=np.int64) for y in BINS}
V = np.stack([np.asarray(sp.mat[y][rows[y]]) for y in BINS], axis=1).astype(np.float32)
obs = dir_pers_of(V)
acc = np.zeros(len(words))
for k in range(a.K):
    perms = np.argsort(rng.random((len(words), len(BINS))), axis=1)
    acc += dir_pers_of(V[np.arange(len(words))[:, None], perms, :])
excess = obs - acc / a.K
lf = np.log10(C.sum(1))
T = pd.DataFrame({"obs": obs, "excess": excess, "log_freq": lf,
                  "msr": msr, "edge": edge}, index=words)

med = T.excess.median()
dec = pd.qcut(T.log_freq, 10, labels=False, duplicates="drop")
dm = [T.excess[dec == i].median() for i in range(10)]
print(f"\nobserved dir_pers median {T.obs.median():+.4f}")
print(f"EXCESS median {med:+.4f}   % words > 0: {(T.excess > 0).mean() * 100:.1f}%")
print("  by freq decile (rare->frequent): " + " ".join(f"{d:+.3f}" for d in dm))
print(f"  excess x edge-thinness (freq-partial): "
      f"{partial(T.excess, T.edge, T.log_freq):+.4f}   "
      f"[Reddit real -0.059, shuffled -0.061 - if this is near 0, the "
      f"mechanical channel is absent, not merely controlled]")

# ---- 3. the content leg, human predictor -----------------------------------
brys = pd.read_csv(os.path.expanduser(a.brysbaert), sep="\t")
brys["Word"] = brys.Word.astype(str).str.lower()
conc = brys[~brys.Word.str.contains(" ")].set_index("Word")["Conc.M"]
dw, dv = sp.drift(BINS[0], BINS[-1])
T["dr"] = pd.Series(dict(zip(dw, dv))).reindex(T.index)
X = T.join(conc.rename("conc"), how="inner").dropna(subset=["excess", "conc", "log_freq", "dr"])
q = fm_pct(X.dr.to_numpy(), X.log_freq.to_numpy())
bands = [("all", 0.0), ("top40%", 0.6), ("top20%", 0.8), ("top10%", 0.9)]
res = {lab: partial(X[q >= b].excess, X[q >= b].conc, X[q >= b].log_freq) for lab, b in bands}
print(f"\nrho(brysbaert, excess | freq) by drift band   n={len(X):,}")
print("  " + "   ".join(f"{lab} {v:+.4f}" for lab, v in res.items()))
print("  Reddit 5-sub:  all +0.0216   top40% +0.0663   top20% +0.1019   top10% +0.1427")

vals = [res[lab] for lab, _ in bands]
mono = all(vals[i] <= vals[i + 1] for i in range(3))
verdict = ("REPLICATES" if res["top20%"] >= 0.08 and mono and med >= 0.02
           and sum(d > 0 for d in dm) >= 8
           else "DEAD" if res["top20%"] <= 0.02 or med <= 0.005 else "PARTIAL")
if res["top20%"] <= -0.08:
    verdict = "CONTRADICTS (sign reversed vs Reddit - report as such)"
print(f"\n== PRE-REGISTERED VERDICT: top20% {res['top20%']:+.4f} (bar +0.08), "
      f"monotone {mono}, median excess {med:+.4f} -> {verdict}")

print("\ntop-30 excess (face validity; Reddit surfaced trump/salty/ghost/discord):")
print("  " + " ".join(T.nlargest(30, "excess").index.tolist()))
T.to_csv(a.out)
print(f"\nwrote {a.out}")
