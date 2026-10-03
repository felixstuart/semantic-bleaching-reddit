"""PRE-REGISTERED 3-sub confirmation of the content->orderedness gradient.

Registered 2026-08-24 BEFORE running (bar stated in the prior session message):
  Test: rho(pred_conc, order-permutation excess | log_freq) in fm-drift bands
        (all / top40% / top20% / top10%), on the 3-sub space
        (vectors/sgns.words), K=100 per-word order permutations.
  CONFIRMED   top-20% band rho >= +0.08 AND band rhos non-decreasing
              (tolerance 0.01) across all->top10%.
  FAILED      top-20% band rho <= +0.04.
  else        INTERMEDIATE.

Notes on record: pred_conc comes from the 5-sub context-space imputation
(cross-training-run predictor; the two corpora share 3 subreddits, so this is
replication across training runs + partial data, not fully independent data).
No shuffled arm exists for 3-sub; the estimation-quality rival was ruled out
on 5-sub and the order-permutation null itself is self-contained per word.
Sanity (not verdicts): excess median should be positive with a rising
frequency gradient; known changers should rank high freq-matched.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402

YEARS = list(range(2012, 2019))
K = 100
rng = np.random.default_rng(11)

sp = Space(REPO + "/vectors/sgns.words", cache_dir=REPO + "/cache", verbose=True)
words = [w for w in sp.words[2012] if all(w in sp.index[y] for y in YEARS)]
V = np.stack([np.asarray(sp.mat[y][[sp.index[y][w] for w in words]])
              for y in YEARS], axis=1).astype(np.float32)
n = len(words)
print(f"3-sub complete-case words: {n:,}")


def dir_pers_of(V):
    D = V[:, 1:, :] - V[:, :-1, :]
    Dn = D / (np.linalg.norm(D, axis=2, keepdims=True) + 1e-12)
    return np.einsum("ntd,ntd->nt", Dn[:, 1:, :], Dn[:, :-1, :]).mean(1)


obs = dir_pers_of(V)
acc = np.zeros(n)
for k in range(K):
    perms = np.argsort(rng.random((n, 7)), axis=1)
    acc += dir_pers_of(V[np.arange(n)[:, None], perms, :])
excess = obs - acc / K

counts = sp.counts(REPO + "/counts.words.vocab")
lf = np.array([np.log10(sum(counts.get(w, {}).values())) if counts.get(w) else np.nan
               for w in words])
print(f"vocab coverage of space words: {np.isfinite(lf).mean() * 100:.1f}%")
dw, dv = sp.drift(2012, 2018)
dmap = dict(zip(dw, dv))
P = pd.read_csv(REPO + "/data/tn_validity/predicted_concreteness.csv",
                index_col=0, keep_default_na=False)
T = pd.DataFrame({"excess": excess, "obs": obs, "log_freq": lf}, index=words)
T["drift"] = pd.Series(dmap).reindex(words)
T = T.join(P.pred_conc)

# sanity
S = T.dropna(subset=["excess", "log_freq"])
dec = pd.qcut(S.log_freq, 10, labels=False, duplicates="drop")
print(f"\nsanity: excess median {S.excess.median():+.4f}  ({(S.excess > 0).mean() * 100:.1f}% > 0)")
print("  by freq decile: " + " ".join(f"{S.excess[dec == i].median():+.3f}" for i in range(10)))
o = np.argsort(S.log_freq.values); l2 = S.log_freq.values[o]; e2 = S.excess.values[o]
lo = np.searchsorted(l2, l2 - .25, "left"); hi = np.searchsorted(l2, l2 + .25, "right")
fm = np.array([(e2[lo[i]:hi[i]] < e2[i]).mean() for i in range(len(e2))])
pct = np.empty(len(S)); pct[o] = fm * 100
S = S.assign(excess_pct=pct)
for w in ("fam", "chief", "salty", "ghost", "discord", "spiked", "neutrality"):
    if w in S.index:
        print(f"    {w:>10}: {S.loc[w, 'excess_pct']:5.1f}th pctile")

# pre-registered bands
C = S.dropna(subset=["drift", "pred_conc"])
o = np.argsort(C.log_freq.values); l3 = C.log_freq.values[o]; d3 = C.drift.values[o]
lo = np.searchsorted(l3, l3 - .25, "left"); hi = np.searchsorted(l3, l3 + .25, "right")
fm = np.array([(d3[lo[i]:hi[i]] < d3[i]).mean() for i in range(len(d3))])
dp = np.empty(len(C)); dp[o] = fm * 100
C = C.assign(drift_pct=dp)


def part(sub):
    rx, ry, rz = rankdata(sub.pred_conc), rankdata(sub.excess), rankdata(sub.log_freq)
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1])


print("\n== PRE-REGISTERED BANDS: rho(pred_conc, excess | freq) ==")
bands = []
for label, thr in [("all", 0), ("top 40% drift", 60),
                   ("top 20% drift", 80), ("top 10% drift", 90)]:
    sub = C[C.drift_pct >= thr]
    r = part(sub)
    bands.append(r)
    print(f"  {label:>14}: {r:+.4f}   n={len(sub):,}")
top20 = bands[2]
monotone = all(bands[i + 1] >= bands[i] - 0.01 for i in range(3))
verdict = ("CONFIRMED" if top20 >= 0.08 and monotone
           else "FAILED" if top20 <= 0.04 else "INTERMEDIATE")
print(f"\n  top-20% = {top20:+.4f} (bar +0.08)   monotone(tol .01): {monotone}")
print(f"  VERDICT: {verdict}")
S.to_csv(REPO + "/data/step_profile/orderperm_3sub.csv")
