"""Independent re-verification of the TN + concentration kills.

1. Spot-check cached chaining/TN against raw vectors (validates driftcache).
2. Rebuild the battery table with FRESH code; reproduce the iter15 REAL rows.
3. NEW: run the notebook cell-43 headline gradient (TN x conc_steps within
   fm-drift-percentile bins) on the SHUFFLED space -- never tested before.
4. Recompute the 2026-08-22 anchor-sharing table from scratch.
5. Concentration mechanism checks (CV proxy, scale-invariance simulation).
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = "/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad"
SC10 = "/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-Programming-Club-SemanticDrift/10425496-07a9-47a3-8abd-65f4fccd6a77/scratchpad"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402

sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)

# ---------------------------------------------------------------- 1. spot-check
print("=" * 78)
print("1. SPOT-CHECK: cached chaining vs direct recomputation from raw vectors")
print("=" * 78)
df = sp.chaining()
rng = np.random.default_rng(0)
ok = True
for i in rng.choice(len(df), 8, replace=False):
    row = df.iloc[i]
    w, y = row.word, int(row.year)
    va = np.asarray(sp.mat[y][sp.index[y][w]])
    vb = np.asarray(sp.mat[y + 1][sp.index[y + 1][w]])
    step = 1.0 - float(va @ vb)
    sims = np.asarray(sp.mat[y]) @ va
    sims[sp.index[y][w]] = -np.inf
    tn_direct = 1.0 - float(np.sort(sims)[-25:].mean())
    ds, dt = abs(step - row.step), abs(tn_direct - row.radius)
    ok &= ds < 1e-5 and dt < 1e-5
    print(f"  {w:>16} {y}  step cache {row.step:.6f} direct {step:.6f}   "
          f"TN cache {row.radius:.6f} direct {tn_direct:.6f}")
print("  => cache", "MATCHES raw vectors" if ok else "*** MISMATCH ***")

# ------------------------------------------------- 2. rebuild battery, fresh code
print()
print("=" * 78)
print("2. FRESH-CODE battery on REAL space vs the saved iter15 'real' rows")
print("=" * 78)
g = df.groupby("word")
T = pd.DataFrame({
    "n_trans": g["step"].size(),
    "concentration": g["ratio"].max() / g["ratio"].sum(),
    "conc_steps": g["step"].max() / g["step"].sum(),
    "step_mean": g["step"].mean(),
    "step_sd": g["step"].std(),
})
T["step_cv"] = T.step_sd / T.step_mean
counts = sp.counts(REPO + "/vectors/5sub/counts.words.vocab")
tot = {w: sum(d.values()) for w, d in counts.items()}
T["log_freq"] = np.log10(pd.Series({w: tot.get(w, np.nan) for w in T.index}))
dw, dv = sp.drift(2012, 2018)
T["drift"] = pd.Series(dict(zip(dw, dv)))
T["tn_first"] = pd.Series(sp.tn[2012])
T["tn_mid"] = pd.Series(sp.tn[2015])
T = T[T.n_trans == 6].dropna(subset=["tn_first", "tn_mid", "drift",
                                     "concentration", "conc_steps", "log_freq"])
print(f"  complete-case words: {len(T):,}  (iter15 file said 71,653)")


def partial_rho(x, y, z):
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    rx, ry, rz = (rankdata(v[m]) for v in (x, y, z))
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1]), int(m.sum())


def fm_bins(t, xcol, ycol, nbins=10):
    b = pd.qcut(t.log_freq, nbins, labels=False, duplicates="drop")
    rhos = [spearmanr(x[xcol], x[ycol]).correlation for _, x in t.groupby(b)]
    return rhos


PAIRS = [("tn_first", "concentration", -0.1138, -0.1328, -0.0907),
         ("tn_first", "conc_steps", +0.0500, -0.0226, +0.0117),
         ("tn_mid", "conc_steps", -0.0535, -0.0695, -0.0326),
         ("tn_first", "drift", -0.0144, +0.2746, +0.2076),
         ("concentration", "drift", +0.0051, +0.0389, +0.1288)]
for a, b, exp_o, exp_p, exp_m in PAIRS:
    o = spearmanr(T[a], T[b]).correlation
    p, _ = partial_rho(T[a].to_numpy(float), T[b].to_numpy(float),
                       T.log_freq.to_numpy(float))
    rhos = fm_bins(T, a, b)
    m = float(np.mean(rhos))
    print(f"  {a:>13} x {b:<13} overall {o:+.4f} (file {exp_o:+.4f})   "
          f"partial {p:+.4f} (file {exp_p:+.4f})   fm-mean {m:+.4f} (file {exp_m:+.4f})")
    print(f"                per-bin: " + " ".join(f"{r:+.3f}" for r in rhos))

# ------------------------------------------- 3. cell-43 gradient, real vs shuffled
print()
print("=" * 78)
print("3. NOTEBOOK CELL-43 HEADLINE GRADIENT on real vs SHUFFLED (new test)")
print("   rho(TN, conc_steps) within 20 bins of freq-matched drift percentile")
print("=" * 78)
csv = pd.read_csv(SC6 + "/null_vs_real_table.csv")
print(f"  loaded {len(csv):,} rows: " +
      ", ".join(f"{k}={v:,}" for k, v in csv.space.value_counts().items()))


def cell43(t, label):
    t = t.dropna(subset=["tn_first", "tn_mid", "conc_steps", "drift", "log_freq"])
    lf = t.log_freq.to_numpy(float)
    dr = t.drift.to_numpy(float)
    order = np.argsort(lf)
    lfs, drs = lf[order], dr[order]
    lo = np.searchsorted(lfs, lfs - 0.25, side="left")
    hi = np.searchsorted(lfs, lfs + 0.25, side="right")
    fm = np.empty(len(t))
    for k in range(len(t)):
        fm[k] = (drs[lo[k]:hi[k]] < drs[k]).mean() * 100
    fm_pct = np.empty(len(t))
    fm_pct[order] = fm
    NB = 20
    edges = np.array([np.percentile(fm_pct, (100 / NB) * i) for i in range(1, NB + 1)])
    idx = np.digitize(fm_pct, edges)
    rows = []
    for bb in range(NB):
        m = idx == bb
        if m.sum() < 30:
            continue
        r12 = spearmanr(t.tn_first[m], t.conc_steps[m]).correlation
        r15 = spearmanr(t.tn_mid[m], t.conc_steps[m]).correlation
        rows.append((bb, int(m.sum()), r12, r15,
                     float(np.median(lf[m])), float(np.median(dr[m]))))
    bs = [r[0] for r in rows]
    m12 = spearmanr(bs, [r[2] for r in rows]).correlation
    m15 = spearmanr(bs, [r[3] for r in rows]).correlation
    print(f"\n  [{label}]  n={len(t):,}")
    print(f"    TN2012 x conc_steps: {rows[0][2]:+.3f} -> {rows[-1][2]:+.3f}   "
          f"monotonicity {m12:+.3f}")
    print(f"    TN2015 x conc_steps: {rows[0][3]:+.3f} -> {rows[-1][3]:+.3f}   "
          f"monotonicity {m15:+.3f}")
    print(f"    log-freq flatness: {rows[0][4]:.2f} -> {rows[-1][4]:.2f}")
    print("    per-bin TN2012: " + " ".join(f"{r[2]:+.2f}" for r in rows))
    print("    per-bin TN2015: " + " ".join(f"{r[3]:+.2f}" for r in rows))
    return rows


cell43(csv[csv.space == "real"], "REAL   (notebook said +0.104->-0.220 mono -0.940; "
       "+0.072->-0.277 mono -0.977)")
cell43(csv[csv.space == "shuffled"], "SHUFFLED null")

# --------------------------------------------------- 4. anchor-sharing table
print()
print("=" * 78)
print("4. ANCHOR-SHARING TABLE recomputed from scratch (partial for log_freq)")
print("=" * 78)
S3 = pd.read_pickle(SC10 + "/S3.pkl")
print(f"  S3.pkl columns: {list(S3.columns)}  n={len(S3):,}")
sen = json.load(open(REPO + "/senses.json"))
ns = {k: sum(len(b.get("senses", [])) for b in v) for k, v in sen.items()}
base = T.copy()
base["tn2014"] = pd.Series(sp.tn[2014])
base["ctn50"] = S3["ctn50"] if "ctn50" in S3.columns else np.nan
base["n_senses"] = pd.Series({w: ns.get(w, np.nan) for w in base.index})
base.loc[base.n_senses == 0, "n_senses"] = np.nan
mem = {(2012): (+0.098, +0.084, -0.148, +0.045),
       (2013): (+0.054, +0.088, -0.156, +0.062),
       (2014): (-0.002, +0.035, -0.211, +0.063),
       (2015): (-0.083, -0.066, -0.317, +0.059)}
print(f"  {'anchor':>12} {'tn2012':>16} {'tn2014':>16} {'ctn50':>16} {'n_senses':>16}")
for a in (2012, 2013, 2014, 2015):
    ww, vv = sp.drift(a, 2018)
    d = pd.Series(dict(zip(ww, vv))).reindex(base.index)
    cells = []
    for j, col in enumerate(["tn_first", "tn2014", "ctn50", "n_senses"]):
        r, n = partial_rho(base[col].to_numpy(float), d.to_numpy(float),
                           base.log_freq.to_numpy(float))
        cells.append(f"{r:+.3f} (mem {mem[a][j]:+.3f})")
    print(f"  {a}->2018   " + " ".join(f"{c:>16}" for c in cells))
r, n = partial_rho(base.log_freq.to_numpy(float),
                   base.drift.to_numpy(float), base.drift.to_numpy(float) * 0 + 1)
conf = spearmanr(base.log_freq, base.drift).correlation
print(f"\n  law of conformity rho(log_freq, drift) = {conf:+.3f}  n={len(base):,}  "
      f"(mem -0.752 at n=28,621)")
w1, v1 = sp.drift(2012, 2015)
w2, v2 = sp.drift(2015, 2018)
d1 = pd.Series(dict(zip(w1, v1))).reindex(base.index)
d2 = pd.Series(dict(zip(w2, v2))).reindex(base.index)
raw = spearmanr(d1, d2, nan_policy="omit").correlation
pr, n = partial_rho(d1.to_numpy(float), d2.to_numpy(float),
                    base.log_freq.to_numpy(float))
print(f"  drift persistence d(12-15) x d(15-18): raw {raw:+.3f} partial {pr:+.3f} "
      f"n={n:,}  (mem +0.817 / +0.515)")

# ------------------------------------------------- 5. concentration mechanism
print()
print("=" * 78)
print("5. CONCENTRATION MECHANISM (CV proxy + scale-invariance simulation)")
print("=" * 78)
print(f"  rho(concentration, step_mean) = "
      f"{spearmanr(T.concentration, T.step_mean).correlation:+.3f}  (mem -0.064)")
print(f"  rho(concentration, step_cv)   = "
      f"{spearmanr(T.concentration, T.step_cv, nan_policy='omit').correlation:+.3f}"
      f"  (mem +0.694)")
print(f"  concentration median {T.concentration.median():.4f} (mem 0.2333)   "
      f"step_cv median {T.step_cv.median():.4f} (mem 0.2560)")
rng = np.random.default_rng(1)
for cv in (0.10, 0.20, 0.30, 0.40):
    sig = np.sqrt(np.log(1 + cv ** 2))
    x = rng.lognormal(0, sig, size=(200_000, 6))
    ms = np.median(x.max(1) / x.sum(1))
    print(f"  iid-lognormal CV={cv:.2f}: median max/sum = {ms:.4f}  "
          f"(mem {'0.1872 0.2084 0.2298 0.2508'.split()[int(cv * 10) - 1]})")
