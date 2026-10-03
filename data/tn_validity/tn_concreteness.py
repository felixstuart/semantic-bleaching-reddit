"""Is TN a usable concreteness instrument on Reddit? Full validity battery.

Known going in: anchor-year validity peaks 2014 (~-0.26), ctn50 ~-0.28 and
frequency-flat, Hansard/GB ~0 (corpus-specific), TN2012-vs-2014 rank agreement
only ~0.55. Open questions this script answers:
  1. VALIDITY    per anchor year + 7-year mean, raw and freq-partial.
  2. RELIABILITY inter-year agreement -> Spearman-Brown ceiling for the mean;
                 disattenuated validity (conservatively treating Brysbaert as
                 perfectly reliable).
  3. VARIANTS    k=1/3/5/10/25, median, sim-SD, sim-decay at 2014; context-space
                 ctn25/50/100/200 + tnwa5k/20k from S3.
  4. CONFOUNDS   partial for n_senses, freq+senses+length jointly; validity
                 within POS (Brysbaert Dom_Pos); flatness across freq deciles.
  5. DENSITY     (a) validity within 2014-count bins; (b) NULL-space pseudo-year
                 validity -- pseudo-years differ ONLY in data mass (semantics
                 held constant), so validity-vs-mass falls out directly, and
                 real-vs-null at matched year isolates temporal composition.
  6. CONC.SD     diffuseness vs human rater disagreement (control |conc-3|
                 for the extremity U-shape).
  7. ITEMS       largest residuals (what TN gets wrong); face-validity read of
                 extreme-TN words OUTSIDE Brysbaert coverage (the backfill case).
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

YEARS = list(range(2012, 2019))
sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)

# ---------------------------------------------------------------- frame
b = pd.read_csv(REPO + "/data/brysbaert_concreteness.txt", sep="\t",
                keep_default_na=False)
b = b[(b.Bigram == 0)].copy()
b["Word"] = b.Word.astype(str).str.lower()
b = b.drop_duplicates("Word").set_index("Word")

counts = sp.counts(REPO + "/vectors/5sub/counts.words.vocab")
words = [w for w in sp.words[2012]
         if all(w in sp.index[y] for y in YEARS)]
F = pd.DataFrame(index=words)
for y in YEARS:
    F[f"tn{y}"] = pd.Series(sp.tn[y]).reindex(words)
F["tn_mean"] = F[[f"tn{y}" for y in YEARS]].mean(axis=1)
F["log_freq"] = np.log10([sum(counts.get(w, {}).values()) or np.nan for w in words])
F["c2014"] = [counts.get(w, {}).get(2014, 0) for w in words]
F["length"] = [len(w) for w in words]
sen = json.load(open(REPO + "/senses.json"))
ns = {k: sum(len(x.get("senses", [])) for x in v) for k, v in sen.items()}
F["n_senses"] = pd.Series({w: ns.get(w, np.nan) for w in words})
F.loc[F.n_senses == 0, "n_senses"] = np.nan
S3 = pd.read_pickle(SC10 + "/S3.pkl")
for c in ("ctn25", "ctn50", "ctn100", "ctn200", "tnwa5k", "tnwa20k"):
    F[c] = S3[c].reindex(words)
F["conc"] = b["Conc.M"].reindex(words)
F["conc_sd"] = b["Conc.SD"].reindex(words)
F["pos"] = b["Dom_Pos"].reindex(words)
C = F.dropna(subset=["conc", "log_freq"])
print(f"sample: {len(F):,} words in all 7 years; {len(C):,} Brysbaert-covered")


def partial(x, y, controls):
    cols = [np.asarray(x, float), np.asarray(y, float)] + [np.asarray(c, float) for c in controls]
    m = ~np.any([np.isnan(c) for c in cols], axis=0)
    if m.sum() < 30:
        return np.nan, 0
    r = [rankdata(c[m]) for c in cols]
    Z = np.column_stack([np.ones(m.sum())] + r[2:])
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(r[0]), res(r[1]))[0, 1]), int(m.sum())


# ---------------------------------------------------------------- 1. validity
print("\n== 1. VALIDITY vs Brysbaert Conc.M (raw / freq-partial) ==")
for col in [f"tn{y}" for y in YEARS] + ["tn_mean"]:
    raw = spearmanr(C[col], C.conc, nan_policy="omit").correlation
    pr, n = partial(C[col], C.conc, [C.log_freq])
    print(f"  {col:>8}  raw {raw:+.3f}   partial {pr:+.3f}   n={n:,}")

# ---------------------------------------------------------------- 2. reliability
print("\n== 2. RELIABILITY ==")
tnmat = F[[f"tn{y}" for y in YEARS]]
rhos = []
for i, a in enumerate(YEARS):
    for bb in YEARS[i + 1:]:
        rhos.append(spearmanr(tnmat[f"tn{a}"], tnmat[f"tn{bb}"]).correlation)
rbar = float(np.mean(rhos))
sb7 = 7 * rbar / (1 + 6 * rbar)
print(f"  mean inter-year rho = {rbar:.3f}  (range {min(rhos):.3f}..{max(rhos):.3f})")
print(f"  Spearman-Brown reliability of 7-year mean = {sb7:.3f}")
raw_mean = abs(spearmanr(C.tn_mean, C.conc, nan_policy="omit").correlation)
print(f"  disattenuated |validity| of tn_mean (Brysbaert treated as perfect): "
      f"{raw_mean / np.sqrt(sb7):.3f}")

# ---------------------------------------------------------------- 3. variants
print("\n== 3. VARIANTS at 2014 (raw / freq-partial) ==")
sim14, w14 = sp.knn_sim[2014], sp.words[2014]
variants = {f"tn14_k{k}": dict(zip(w14, (1 - sim14[:, :k].mean(1)).tolist()))
            for k in (1, 3, 5, 10, 25)}
variants["tn14_median"] = dict(zip(w14, (1 - np.median(sim14, 1)).tolist()))
variants["sim14_sd"] = dict(zip(w14, sim14.std(1).tolist()))
variants["sim14_decay"] = dict(zip(w14, (sim14[:, 0] - sim14[:, -1]).tolist()))
for name, d in variants.items():
    v = pd.Series(d).reindex(C.index)
    raw = spearmanr(v, C.conc, nan_policy="omit").correlation
    pr, n = partial(v, C.conc, [C.log_freq])
    print(f"  {name:>12}  raw {raw:+.3f}   partial {pr:+.3f}   n={n:,}")
for col in ("ctn25", "ctn50", "ctn100", "ctn200", "tnwa5k", "tnwa20k"):
    raw = spearmanr(C[col], C.conc, nan_policy="omit").correlation
    pr, n = partial(C[col], C.conc, [C.log_freq])
    print(f"  {col:>12}  raw {raw:+.3f}   partial {pr:+.3f}   n={n:,}")

# ---------------------------------------------------------------- 4. confounds
print("\n== 4. CONFOUNDS (tn2014 and ctn50) ==")
for col in ("tn2014_", "ctn50"):
    v = pd.Series(variants["tn14_k25"]).reindex(C.index) if col == "tn2014_" else C[col]
    p1, n1 = partial(v, C.conc, [C.log_freq])
    p2, n2 = partial(v, C.conc, [C.log_freq, C.n_senses])
    p3, n3 = partial(v, C.conc, [C.log_freq, C.n_senses, C.length])
    print(f"  {col:>8}: partial(freq) {p1:+.3f} n={n1:,}   +senses {p2:+.3f} n={n2:,}"
          f"   +length {p3:+.3f} n={n3:,}")
v14 = pd.Series(variants["tn14_k25"]).reindex(C.index)
print("  within-POS validity (freq-partial):")
for pos in ("Noun", "Adjective", "Verb", "Adverb"):
    sub = C[C.pos == pos]
    pr, n = partial(v14.loc[sub.index], sub.conc, [sub.log_freq])
    pc, _ = partial(sub.ctn50, sub.conc, [sub.log_freq])
    print(f"    {pos:>10}  tn2014 {pr:+.3f}   ctn50 {pc:+.3f}   n={n:,}")
print("  flatness across freq deciles (raw rho within decile):")
dec = pd.qcut(C.log_freq, 10, labels=False, duplicates="drop")
r14 = [spearmanr(v14[dec == i], C.conc[dec == i], nan_policy="omit").correlation for i in range(10)]
r50 = [spearmanr(C.ctn50[dec == i], C.conc[dec == i], nan_policy="omit").correlation for i in range(10)]
print("    tn2014: " + " ".join(f"{r:+.2f}" for r in r14))
print("    ctn50 : " + " ".join(f"{r:+.2f}" for r in r50))

# ---------------------------------------------------------------- 5. density
print("\n== 5. DENSITY: does validity scale with data mass? ==")
cb = pd.cut(np.log10(C.c2014.replace(0, np.nan)), bins=[1, 2, 2.5, 3, 3.5, 4, 8])
print("  validity (raw) within 2014-count bins:")
for iv, sub in C.groupby(cb, observed=True):
    if len(sub) < 200:
        continue
    r = spearmanr(v14.loc[sub.index], sub.conc, nan_policy="omit").correlation
    print(f"    count 10^{iv}  rho {r:+.3f}   n={len(sub):,}")
sp_null = Space(SC6 + "/nullvec/null_iter15.words", cache_dir=SC6 + "/cache_null",
                verbose=False)
print("  NULL-space pseudo-year TN validity (mass-only differences, freq-partial):")
mass = {2012: .0625, 2013: .1519, 2014: .1758, 2015: .1683,
        2016: .1619, 2017: .1445, 2018: .1351}
for y in YEARS:
    vn = pd.Series(sp_null.tn[y]).reindex(C.index)
    pr, n = partial(vn, C.conc, [C.log_freq])
    rr = spearmanr(vn, C.conc, nan_policy="omit").correlation
    print(f"    null tn{y} (mass {mass[y]:.3f})  raw {rr:+.3f}  partial {pr:+.3f}  n={n:,}")

# ---------------------------------------------------------------- 6. Conc.SD
print("\n== 6. TN vs human rater disagreement (Conc.SD) ==")
ext = (C.conc - 3.0).abs()
p_sd, n = partial(v14, C.conc_sd, [C.log_freq, ext])
raw_sd = spearmanr(v14, C.conc_sd, nan_policy="omit").correlation
print(f"  tn2014 x Conc.SD  raw {raw_sd:+.3f}   partial(freq, |conc-3|) {p_sd:+.3f}   n={n:,}")

# ---------------------------------------------------------------- 7. items
print("\n== 7. ITEM-LEVEL ==")
m = v14.notna() & C.conc.notna()
rk_tn, rk_c = rankdata(v14[m]), rankdata(C.conc[m])
resid = (rk_tn / m.sum()) + (rk_c / m.sum()) - 1   # high = diffuse AND concrete
idx = C.index[m]
srt = np.argsort(resid)
print("  TN says diffuse, humans say concrete (top-15 misses):")
print("    " + " ".join(idx[srt[-15:]][::-1]))
print("  TN says tight, humans say abstract (top-15 misses):")
inv = (1 - rk_tn / m.sum()) + (rk_c / m.sum() - 0)   # low tn, low conc -> small
srt2 = np.argsort((-(rk_tn / m.sum())) + (-(1 - rk_c / m.sum())))
lo_tn_abstract = np.argsort((rk_tn / m.sum()) + (1 - rk_c / m.sum()))
print("    " + " ".join(idx[lo_tn_abstract[:15]]))
un = F[F.conc.isna() & (F.log_freq >= 3.5)].copy()
un["v14"] = pd.Series(variants["tn14_k25"]).reindex(un.index)
un = un.dropna(subset=["v14"])
print(f"  BACKFILL face read ({len(un):,} uncovered words, freq>=10^3.5):")
print("    tightest (predict concrete): " + " ".join(un.nsmallest(15, "v14").index))
print("    most diffuse (predict abstract): " + " ".join(un.nlargest(15, "v14").index))
