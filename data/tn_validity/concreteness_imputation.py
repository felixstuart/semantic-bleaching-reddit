"""Can concreteness be IMPUTED from the embedding, calibrated on Brysbaert?

Contrast: TN-alone (calibration cannot exceed its rank validity, ~0.26) vs
ridge regression on the full 100-d CONTEXT vector (anchor-free, full-mass).
5-fold CV over WORDS (never over ratings), pure-numpy ridge.

Diagnostics: is the prediction secretly frequency (rho vs log_freq, validity
within freq deciles), within-noun validity, and face reads of the known
failure modes (proper nouns, slang backfill tails).
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402

rng = np.random.default_rng(42)

# ---------------- context vectors (shared across years -> anchor-free)
words, rows = [], []
with open(REPO + "/vectors/5sub/sgns.contexts") as f:
    f.readline()
    for line in f:
        a, b = line.split(" ", 1)
        words.append(a)
        rows.append(np.fromstring(b, dtype=np.float32, sep=" "))
M = np.vstack(rows)
M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
widx = {w: i for i, w in enumerate(words)}
print(f"context matrix {M.shape}")

# ---------------- Brysbaert + covariates
b = pd.read_csv(REPO + "/data/brysbaert_concreteness.txt", sep="\t",
                keep_default_na=False)
b = b[b.Bigram == 0].copy()
b["Word"] = b.Word.astype(str).str.lower()
b = b.drop_duplicates("Word").set_index("Word")
sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
counts = sp.counts(REPO + "/vectors/5sub/counts.words.vocab")
tot = {w: sum(d.values()) for w, d in counts.items()}

cov = [w for w in words if w in b.index and tot.get(w, 0) > 0]
y = b["Conc.M"].reindex(cov).to_numpy(float)
X = M[[widx[w] for w in cov]]
lf = np.log10([tot[w] for w in cov])
pos = b["Dom_Pos"].reindex(cov)
print(f"covered training words: {len(cov):,}")

# ---------------- 5-fold CV ridge (features: 100-d vector only, no freq)
folds = rng.permutation(len(cov)) % 5
alphas = [0.1, 1.0, 10.0, 100.0]
pred = {a: np.empty(len(cov)) for a in alphas}
for k in range(5):
    tr, te = folds != k, folds == k
    Xtr, ytr = X[tr], y[tr]
    mu = ytr.mean()
    G = Xtr.T @ Xtr
    for a in alphas:
        wgt = np.linalg.solve(G + a * np.eye(X.shape[1]), Xtr.T @ (ytr - mu))
        pred[a][te] = X[te] @ wgt + mu
print("\n== CV validity (Spearman pred vs held-out Conc.M) ==")
best_a, best_r = None, -1
for a in alphas:
    r = spearmanr(pred[a], y).correlation
    print(f"  ridge alpha={a:<6} rho = {r:+.4f}")
    if r > best_r:
        best_a, best_r = a, r
P = pred[best_a]
print(f"  (TN-alone ceiling for comparison: 0.26; ctn50-alone: 0.24)")

# ---------------- diagnostics
print("\n== diagnostics ==")
print(f"  rho(pred, log_freq) = {spearmanr(P, lf).correlation:+.4f}   "
      f"(rho(Conc.M, log_freq) = {spearmanr(y, lf).correlation:+.4f})")
dec = pd.qcut(lf, 10, labels=False, duplicates="drop")
rr = [spearmanr(P[dec == i], y[dec == i]).correlation for i in range(10)]
print("  validity within freq deciles: " + " ".join(f"{r:+.2f}" for r in rr))
for p in ("Noun", "Adjective", "Verb", "Adverb"):
    m = (pos == p).to_numpy()
    print(f"  within {p:<10} rho = {spearmanr(P[m], y[m]).correlation:+.4f}  n={m.sum():,}")
res = rankdata(P) / len(P) - rankdata(y) / len(P)
o = np.argsort(res)
cova = np.array(cov)
print("  worst over-predictions (model says concrete, raters abstract): "
      + " ".join(cova[o[-10:]][::-1]))
print("  worst under-predictions (model says abstract, raters concrete): "
      + " ".join(cova[o[:10]]))

# ---------------- refit on all covered words, impute the uncovered
Gf = X.T @ X
wgt = np.linalg.solve(Gf + best_a * np.eye(X.shape[1]), X.T @ (y - y.mean()))
unc = [w for w in words if w not in b.index and tot.get(w, 0) >= 3000]
Pu = M[[widx[w] for w in unc]] @ wgt + y.mean()
U = pd.Series(Pu, index=unc).sort_values()
print(f"\n== backfill: {len(unc):,} uncovered words (freq>=3000) ==")
print("  predicted MOST concrete: " + " ".join(U.index[-18:][::-1]))
print("  predicted MOST abstract: " + " ".join(U.index[:18]))
for probe in ("fam", "bruh", "lowkey", "salty", "yeet", "doggo", "juul",
              "tbh", "ngl", "hmu", "istg", "smh"):
    if probe in U.index:
        print(f"    {probe:>8}: predicted {U[probe]:.2f}  "
              f"(scale 1=abstract..5=concrete)")
out = pd.DataFrame({"pred_conc": np.concatenate([P, Pu])},
                   index=list(cova) + unc)
out["covered"] = [1] * len(cova) + [0] * len(unc)
out.to_csv(REPO + "/data/tn_validity/predicted_concreteness.csv")
print(f"\nwrote predicted_concreteness.csv ({len(out):,} words, alpha={best_a})")
