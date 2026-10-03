"""Audit of the +0.2032 high-drift-band content effect.

The band split has never been tested against the null space. Four checks:
  1. NULL-SPACE REDRAW - run the whole gradient inside the pseudo-year space.
     Content cannot predict trajectory there (no temporal content exists), so
     ANY gradient is manufactured by the conditioning structure.
  2. ENTANGLEMENT - drift and excess are both functions of the same 7 vectors.
     If they are geometrically coupled, banding on drift is banding on excess
     and the correlation is a collider artifact. Measured in the null space,
     where the coupling can only be geometric.
  3. FIRST-YEAR SHARE - high-drift words are thin words; the mechanical floor
     scales with thinness. Never applied to a band-conditioned estimate.
  4. FLOOR - report 50 and 1200, and characterise what floor 1200 discards.
"""
import re, os, sys, collections
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

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
        if len(a) == 2 and PAT.match(a[0]):
            m = PAT.match(a[0]); per[m.group(1)][int(m.group(2))] = int(a[1])
    return per


def partial(x, y, z):
    m = ~(pd.isna(x) | pd.isna(y) | pd.isna(z))
    rx, ry, rz = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1]), int(m.sum())


def fm_pct(v, lf, dex=0.25):
    o = np.argsort(lf); l2, v2 = lf[o], v[o]
    lo = np.searchsorted(l2, l2 - dex, "left"); hi = np.searchsorted(l2, l2 + dex, "right")
    fm = np.array([(v2[lo[i]:hi[i]] < v2[i]).mean() for i in range(len(v2))])
    r = np.empty(len(v2)); r[o] = fm
    return r


brys = pd.read_csv(REPO + "/data/brysbaert_concreteness.txt", sep="\t")
brys["Word"] = brys.Word.astype(str).str.lower()
conc = brys[~brys.Word.str.contains(" ")].drop_duplicates("Word").set_index("Word")["Conc.M"]

frames = {}
for arm, vec, cache, cvocab, csv in (
        ("real", REPO + "/vectors/5sub/sgns.words", REPO + "/cache",
         REPO + "/vectors/5sub/counts.words.vocab", "orderperm_real.csv"),
        ("null", SC6 + "/nullvec/null_iter15.words", SC6 + "/cache_null",
         None, "orderperm_shuffled.csv")):
    sp = Space(vec, cache_dir=cache, verbose=False)
    T = pd.read_csv(REPO + "/data/step_profile/" + csv, index_col=0, keep_default_na=False)
    dw, dv = sp.drift(2012, 2018)
    T["drift"] = pd.Series(dict(zip(dw, dv))).reindex(T.index)
    if cvocab and os.path.exists(cvocab):
        C = load_counts(cvocab)
        M = np.array([[C.get(w, {}).get(y, 0) for y in YEARS] for w in T.index], float)
        share = M.sum(0) / M.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            T["fy_share"] = M[:, 0] / M.sum(1) / share[0]
            T["min_n"] = M.min(1)
    else:   # null counts vocab was reaped from /private/tmp; only the real arm needs it
        T["fy_share"] = np.nan; T["min_n"] = np.nan
    T["conc"] = conc.reindex(T.index)
    frames[arm] = T.dropna(subset=["excess", "drift", "log_freq"])
    print(f"{arm}: {len(frames[arm]):,} words")

BANDS = [("all", 0.0), ("top40%", 0.6), ("top20%", 0.8), ("top10%", 0.9)]
print("\n" + "=" * 74)
print("1. NULL-SPACE REDRAW  rho(brysbaert, excess | freq) by freq-matched drift band")
print("=" * 74)
print(f"{'arm':<8}{'n':>8}" + "".join(f"{lab:>11}" for lab, _ in BANDS))
for arm in ("real", "null"):
    X = frames[arm].dropna(subset=["conc"]).copy()
    X["p"] = fm_pct(X.drift.to_numpy(), X.log_freq.to_numpy())
    row = [partial(X[X.p >= b].excess.to_numpy(), X[X.p >= b].conc.to_numpy(),
                   X[X.p >= b].log_freq.to_numpy())[0] for _, b in BANDS]
    print(f"{arm:<8}{len(X):>8,}" + "".join(f"{v:>+11.4f}" for v in row))
print("  -> a gradient in the null arm = the band split manufactures it")

print("\n" + "=" * 74)
print("2. ENTANGLEMENT of drift and excess (both from the same 7 vectors)")
print("=" * 74)
for arm in ("real", "null"):
    T = frames[arm]
    r0 = spearmanr(T.drift, T.excess).correlation
    r1, n_ = partial(T.excess.to_numpy(), T.drift.to_numpy(), T.log_freq.to_numpy())
    print(f"  {arm:<6} rho(drift, excess) raw {r0:+.4f}   freq-partialled {r1:+.4f}   n={n_:,}")
print("  null arm has NO temporal content, so its coupling is purely geometric")

print("\n" + "=" * 74)
print("3. FIRST-YEAR SHARE inside the high-drift band (real arm)")
print("=" * 74)
X = frames["real"].dropna(subset=["conc", "fy_share"]).copy()
X["p"] = fm_pct(X.drift.to_numpy(), X.log_freq.to_numpy())
H = X[X.p >= 0.8].copy()
print(f"  fy_share median: high-drift band {H.fy_share.median():.3f}  "
      f"all words {X.fy_share.median():.3f}")
H["q"] = pd.qcut(H.fy_share, 4, labels=False, duplicates="drop")
for q in range(4):
    s = H[H.q == q]
    r, n_ = partial(s.excess.to_numpy(), s.conc.to_numpy(), s.log_freq.to_numpy())
    print(f"    fy quartile {q} (median {s.fy_share.median():.2f}): {r:+.4f}  n={n_:,}")
r, n_ = partial(H.excess.to_numpy(), H.conc.to_numpy(), H.log_freq.to_numpy())
print(f"    unstratified: {r:+.4f}  n={n_:,}")

print("\n" + "=" * 74)
print("4. FLOOR: both, and what floor 1200 discards")
print("=" * 74)
for floor in (0, 1200):
    s = X[X.min_n >= floor].copy()
    s["p"] = fm_pct(s.drift.to_numpy(), s.log_freq.to_numpy())
    row = [partial(s[s.p >= b].excess.to_numpy(), s[s.p >= b].conc.to_numpy(),
                   s[s.p >= b].log_freq.to_numpy())[0] for _, b in BANDS]
    print(f"  floor {floor:>5}  n={len(s):>7,}" + "".join(f"{v:>+11.4f}" for v in row))
lost = X[X.min_n < 1200]; kept = X[X.min_n >= 1200]
print(f"\n  discarded by floor 1200: {len(lost):,} words ({len(lost)/len(X):.1%})")
print(f"    median log_freq   lost {lost.log_freq.median():.2f}  kept {kept.log_freq.median():.2f}")
print(f"    median concreteness lost {lost.conc.median():.3f}  kept {kept.conc.median():.3f}")
print(f"    median drift      lost {lost.drift.median():.4f}  kept {kept.drift.median():.4f}")
print(f"    share of the top-20% drift band that is discarded: "
      f"{(X[X.p >= 0.8].min_n < 1200).mean():.1%}")
