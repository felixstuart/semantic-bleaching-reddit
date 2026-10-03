"""Magnitude-carrying trajectory-shape statistics vs the noise floor.

The ratio family died because it divides out magnitude. These candidates KEEP
magnitude; shape information is extracted by MATCHING (real vs null words
compared within cells of frequency x total path length), never by division.

Candidates (per word, 6 steps, complete-case all-7-years words):
  max_step   largest single-year step
  step_sd    SD of the six steps (magnitude-carrying dispersion)
  spike      max_step - median(steps)  (spike height above typical step)
  gap12      max_step - second-largest (is the jump singular?)
  dir_pers   mean cos(consecutive displacement vectors), 5 terms/word.
             Mechanically NEGATIVE under noise (consecutive displacements share
             the middle vector with opposite signs -> expected corr -0.5 under
             iid); genuine directional change should make it LESS negative.
             The empirical null absorbs the mechanical part.

PRE-REGISTERED VERDICTS (fixed 2026-08-24 before computing):
  Spike family, within pooled (freq decile x L decile) cells having >=100
  words of each space; standardized gap = (median_real - median_null) /
  pooled within-cell IQR:
    DEAD          |median-across-cells gap| < 0.05, or sign consistent in
                  <70% of cells
    SURVIVES      |gap| >= 0.15 and same sign in >=80% of cells
    else INTERMEDIATE
  dir_pers, within frequency deciles (it has no L coupling to match away):
    real-minus-null mean-cos gap: SURVIVES if >= +0.05 with the same sign in
    >=8/10 deciles; DEAD if <= +0.02; else INTERMEDIATE.
  Content leg (rho(pred_conc, stat), freq-partial, plus null reference) is
  computed ONLY for statistics that do not come back DEAD.

Honest prior, on record: the concentration post-mortem (step CV null-identical)
predicts the spike family dies in matched cells; dir_pers is the open one
(step_chain's stable ~19% residual points the same way).
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


def shape_table(sp, counts_path):
    df = sp.chaining().sort_values(["word", "year"])
    n = df.groupby("word")["step"].size()
    keep = set(n[n == 6].index)
    sub = df[df.word.isin(keep)]
    words = sub.word.to_numpy()[::6]
    S = sub.step.to_numpy().reshape(-1, 6)
    srt = np.sort(S, axis=1)
    T = pd.DataFrame(index=words)
    T["L"] = S.sum(1)
    T["max_step"] = srt[:, -1]
    T["step_sd"] = S.std(1, ddof=1)
    T["spike"] = srt[:, -1] - np.median(S, axis=1)
    T["gap12"] = srt[:, -1] - srt[:, -2]
    counts = sp.counts(counts_path)
    T["log_freq"] = np.log10([sum(counts.get(w, {}).values()) or np.nan for w in words])
    ok = [w for w in words if all(w in sp.index[y] for y in YEARS)]
    V = np.stack([np.asarray(sp.mat[y][[sp.index[y][w] for w in ok]]) for y in YEARS], axis=1)
    D = V[:, 1:, :] - V[:, :-1, :]
    Dn = D / (np.linalg.norm(D, axis=2, keepdims=True) + 1e-12)
    ac = np.einsum("ntd,ntd->nt", Dn[:, 1:, :], Dn[:, :-1, :]).mean(1)
    T["dir_pers"] = pd.Series(ac, index=ok)
    return T.dropna(subset=["log_freq"])


print("building real + null shape tables ...", flush=True)
sp_r = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
sp_n = Space(SC6 + "/nullvec/null_iter15.words", cache_dir=SC6 + "/cache_null", verbose=False)
R = shape_table(sp_r, REPO + "/vectors/5sub/counts.words.vocab")
N = shape_table(sp_n, SC6 + "/nullvec/counts.words.vocab")
print(f"real n={len(R):,}   null n={len(N):,}")

print("\n== unconditional distributions (magnitude expected to differ) ==")
print(f"  {'stat':<10} {'real med':>10} {'null med':>10} {'ratio n/r':>10}")
for c in ("L", "max_step", "step_sd", "spike", "gap12", "dir_pers"):
    print(f"  {c:<10} {R[c].median():>10.4f} {N[c].median():>10.4f} "
          f"{N[c].median() / R[c].median() if R[c].median() else float('nan'):>10.3f}")

# ---------------- matched-cell comparison for the spike family
print("\n== PRE-REGISTERED: spike family at matched (freq decile x L decile) ==")
pool_L = pd.concat([R.L, N.L])
pool_f = pd.concat([R.log_freq, N.log_freq])
Ledges = np.quantile(pool_L, np.linspace(0, 1, 11))
fedges = np.quantile(pool_f, np.linspace(0, 1, 11))
for T in (R, N):
    T["Lb"] = np.clip(np.digitize(T.L, Ledges[1:-1]), 0, 9)
    T["fb"] = np.clip(np.digitize(T.log_freq, fedges[1:-1]), 0, 9)
for c in ("max_step", "step_sd", "spike", "gap12"):
    gaps = []
    for lb in range(10):
        for fb in range(10):
            r = R[(R.Lb == lb) & (R.fb == fb)][c]
            m = N[(N.Lb == lb) & (N.fb == fb)][c]
            if len(r) < 100 or len(m) < 100:
                continue
            pooled = pd.concat([r, m])
            iqr = pooled.quantile(.75) - pooled.quantile(.25)
            if iqr <= 0:
                continue
            gaps.append((r.median() - m.median()) / iqr)
    gaps = np.array(gaps)
    med = float(np.median(gaps))
    consist = max((gaps > 0).mean(), (gaps < 0).mean())
    verdict = ("DEAD" if abs(med) < 0.05 or consist < 0.70
               else "SURVIVES" if abs(med) >= 0.15 and consist >= 0.80
               else "INTERMEDIATE")
    print(f"  {c:<10} cells={len(gaps):3d}  median std-gap {med:+.4f}  "
          f"sign-consistency {consist:.2f}  -> {verdict}")

# ---------------- dir_pers within frequency deciles
print("\n== PRE-REGISTERED: dir_pers (real - null mean cos, per freq decile) ==")
gaps = []
for fb in range(10):
    r, m = R[R.fb == fb].dir_pers.dropna(), N[N.fb == fb].dir_pers.dropna()
    if len(r) < 100 or len(m) < 100:
        continue
    gaps.append(r.mean() - m.mean())
    print(f"  decile {fb}: real {r.mean():+.4f}  null {m.mean():+.4f}  "
          f"gap {gaps[-1]:+.4f}  (n={len(r):,}/{len(m):,})")
gaps = np.array(gaps)
same = max((gaps > 0).sum(), (gaps < 0).sum())
g = float(np.median(gaps))
verdict = ("SURVIVES" if g >= 0.05 and same >= 8 else
           "DEAD" if g <= 0.02 else "INTERMEDIATE")
print(f"  median gap {g:+.4f}, consistent in {same}/{len(gaps)} deciles -> {verdict}")
print(f"  overall: real {R.dir_pers.mean():+.4f}  null {N.dir_pers.mean():+.4f}  "
      f"rho(dir_pers, log_freq) real "
      f"{spearmanr(R.dir_pers, R.log_freq, nan_policy='omit').correlation:+.3f} null "
      f"{spearmanr(N.dir_pers, N.log_freq, nan_policy='omit').correlation:+.3f}")

# ---------------- content leg for non-DEAD statistics
surv = []
if verdict != "DEAD":
    surv.append("dir_pers")
if surv:
    print("\n== content leg (only for non-DEAD stats) ==")
    P = pd.read_csv(REPO + "/data/tn_validity/predicted_concreteness.csv",
                    index_col=0, keep_default_na=False)
    for c in surv:
        J = R.join(P.pred_conc).dropna(subset=[c, "pred_conc", "log_freq"])
        rx, ry, rz = rankdata(J[c]), rankdata(J.pred_conc), rankdata(J.log_freq)
        f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
        pr = float(np.corrcoef(f(rx), f(ry))[0, 1])
        Jn = N.join(P.pred_conc).dropna(subset=[c, "pred_conc", "log_freq"])
        rx2, ry2, rz2 = rankdata(Jn[c]), rankdata(Jn.pred_conc), rankdata(Jn.log_freq)
        f2 = lambda a: a - np.polyval(np.polyfit(rz2, a, 1), rz2)  # noqa: E731
        prn = float(np.corrcoef(f2(rx2), f2(ry2))[0, 1])
        print(f"  pred_conc x {c}: real partial {pr:+.4f} (n={len(J):,})   "
              f"NULL reference {prn:+.4f} (n={len(Jn):,})")
R.to_csv(REPO + "/data/step_profile/shape_real.csv")
N.to_csv(REPO + "/data/step_profile/shape_null.csv")
print("\nwrote shape_real.csv / shape_null.csv")
