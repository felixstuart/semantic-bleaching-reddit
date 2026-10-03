"""Within-word ORDER-permutation null for dir_pers.

Per word: dir_pers of the OBSERVED year order vs K=100 random orderings of the
same 7 vectors. The permutation keeps the word's own noise level and the
mechanical ~-0.5 exactly; it destroys only temporal ordering. excess =
observed - mean(permuted). Run identically on the real space and the
time-shuffled space; the shuffled space's pseudo-years carry no order, so its
excess must be ~0 (falsification arm).

Interpretation caveat, on record: positive excess = temporally ORDERED
movement, which includes gradual corpus-composition drift, not only lexical
semantic change. The shuffled arm controls estimation noise, not composition.

PRE-REGISTERED VERDICTS (fixed 2026-08-24 BEFORE running):
  SURVIVES      real median excess >= +0.02, positive in >=8/10 freq deciles,
                AND shuffled-arm median excess <= 50% of real median.
  DEAD          real median excess <= +0.005, OR shuffled/real ratio >= 0.9.
  else          INTERMEDIATE.
  Content leg (rho(pred_conc, excess), freq-partial, shuffled reference)
  computed only if not DEAD.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = "/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402

YEARS = list(range(2012, 2019))
K = 100
rng = np.random.default_rng(7)


def dir_pers_of(V):
    D = V[:, 1:, :] - V[:, :-1, :]
    Dn = D / (np.linalg.norm(D, axis=2, keepdims=True) + 1e-12)
    return np.einsum("ntd,ntd->nt", Dn[:, 1:, :], Dn[:, :-1, :]).mean(1)


def run(sp, vocab, label):
    words = [w for w in sp.words[2012] if all(w in sp.index[y] for y in YEARS)]
    V = np.stack([np.asarray(sp.mat[y][[sp.index[y][w] for w in words]])
                  for y in YEARS], axis=1).astype(np.float32)
    n = len(words)
    obs = dir_pers_of(V)
    acc = np.zeros(n)
    for k in range(K):
        perms = np.argsort(rng.random((n, 7)), axis=1)
        acc += dir_pers_of(V[np.arange(n)[:, None], perms, :])
    excess = obs - acc / K
    counts = sp.counts(vocab)
    lf = np.log10([sum(counts.get(w, {}).values()) or np.nan for w in words])
    T = pd.DataFrame({"obs": obs, "excess": excess, "log_freq": lf}, index=words).dropna()
    med = T.excess.median()
    dec = pd.qcut(T.log_freq, 10, labels=False, duplicates="drop")
    dm = [T.excess[dec == i].median() for i in range(10)]
    print(f"\n[{label}] n={len(T):,}  K={K}")
    print(f"  observed dir_pers median {T.obs.median():+.4f}")
    print(f"  EXCESS median {med:+.4f}   mean {T.excess.mean():+.4f}   "
          f"% words > 0: {(T.excess > 0).mean() * 100:.1f}%")
    print("  median excess by freq decile: " + " ".join(f"{d:+.3f}" for d in dm))
    return T, med, dm


sp_r = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
sp_n = Space(SC6 + "/nullvec/null_iter15.words", cache_dir=SC6 + "/cache_null", verbose=False)
TR, med_r, dm_r = run(sp_r, REPO + "/vectors/5sub/counts.words.vocab", "REAL")
TN_, med_n, dm_n = run(sp_n, SC6 + "/nullvec/counts.words.vocab", "SHUFFLED (falsification arm)")

pos_dec = sum(d > 0 for d in dm_r)
ratio = med_n / med_r if med_r else float("inf")
print("\n== PRE-REGISTERED VERDICT ==")
print(f"  real median excess {med_r:+.4f} (bar +0.02)   positive deciles {pos_dec}/10 (bar 8)")
print(f"  shuffled/real ratio {ratio:.2f} (bar <=0.50 to survive, >=0.90 dead)")
if med_r >= 0.02 and pos_dec >= 8 and ratio <= 0.50:
    verdict = "SURVIVES"
elif med_r <= 0.005 or ratio >= 0.90:
    verdict = "DEAD"
else:
    verdict = "INTERMEDIATE"
print(f"  VERDICT: {verdict}")

if verdict != "DEAD":
    P = pd.read_csv(REPO + "/data/tn_validity/predicted_concreteness.csv",
                    index_col=0, keep_default_na=False)
    for label, T in (("real", TR), ("shuffled", TN_)):
        J = T.join(P.pred_conc).dropna(subset=["excess", "pred_conc", "log_freq"])
        rx, ry, rz = rankdata(J.excess), rankdata(J.pred_conc), rankdata(J.log_freq)
        f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
        print(f"  content leg [{label}]: rho(pred_conc, excess | freq) = "
              f"{float(np.corrcoef(f(rx), f(ry))[0, 1]):+.4f}  n={len(J):,}")
    dw, dv = sp_r.drift(2012, 2018)
    d = pd.Series(dict(zip(dw, dv))).reindex(TR.index)
    rx, ry, rz = rankdata(TR.excess), rankdata(d.fillna(d.median())), rankdata(TR.log_freq)
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    print(f"  excess x drift (real, freq-partial): {float(np.corrcoef(f(rx), f(ry))[0, 1]):+.4f}")
    top = TR.nlargest(30, "excess").index.tolist()
    print("  top-30 excess words (face validity): " + " ".join(top))
TR.to_csv(REPO + "/data/step_profile/orderperm_real.csv")
TN_.to_csv(REPO + "/data/step_profile/orderperm_shuffled.csv")
