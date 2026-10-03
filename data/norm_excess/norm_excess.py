"""Vector-norm bleaching candidate: per-word mechanical-mass correction.

The raw result (2026-08-21) was sd(|dnorm|_null)/sd(|dnorm|_real) = 0.693,
inside the pre-registered <=0.70 bar -- but the null's pseudo-2018 carries 2.2x
pseudo-2012's data, so part of the null's norm GROWTH is mechanical (more SGD
updates -> larger norms), which inflates the real-vs-null contrast. The fix
exploits the null as a calibration set: under shuffled time, norm is a purely
mechanical function of token count plus noise, so fit f(log10 count) -> norm on
the null and subtract each word's f-predicted change computed from its OWN
yearly counts.

PRE-REGISTERED VERDICTS (fixed 2026-08-24, before the corrected statistic was
ever computed; same conventions as compare_null.py):

  GATE      reproduce the recorded raw numbers (norm medians real 3.126/3.103,
            null 3.113/3.269; |dnorm| mean ratio 0.811, sd ratio 0.693) within
            ~+-0.03 or stop -- no verdicts on unverified inputs.
  MODEL     f = binned-mean norm vs log10(count), fit on a deterministic HALF A
            of null words, pooling all 7 pseudo-years; evaluated by interp.
  FLOOR     sd(excess_null on held-out HALF B) / sd(excess_real):
            <=0.70 SURVIVES, >=0.90 DEAD, between -> INTERMEDIATE.
            Secondary: same ratio on mean |excess|.
  FREQ      Spearman(excess_real, log total freq); |rho|<=0.15 = acceptably
            clean (raw real dnorm was -0.069; raw null -0.572).
  VALIDITY  Spearman(freq-residualized mean real norm, Brysbaert Conc.M),
            prediction POSITIVE (Schakel & Wilson 2015: at fixed frequency,
            norm tracks context specificity). Independent of the floor.
  EXPLORE   only if FLOOR is not DEAD: excess x drift (fm-decile mean),
            excess x n_senses partial(freq). Labelled exploratory.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = "/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402

YEARS = list(range(2012, 2019))


def parse_norms(path):
    out = {}
    with open(path) as f:
        f.readline()
        for line in f:
            key, rest = line.split(" ", 1)
            base, y = key.rsplit("_", 1)
            v = np.fromstring(rest, dtype=np.float32, sep=" ")
            out[(base, int(y))] = float(np.linalg.norm(v))
    return out


def parse_counts(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2 or "_" not in parts[0]:
                continue
            base, y = parts[0].rsplit("_", 1)
            try:
                out[(base, int(y))] = int(parts[1])
            except ValueError:
                continue
    return out


def half(word):  # deterministic word-level split
    return int(hashlib.sha1(word.encode()).hexdigest(), 16) % 2


print("parsing norms (2 x ~520k vectors) ...", flush=True)
nr = parse_norms(REPO + "/vectors/5sub/sgns.words")
nn = parse_norms(SC6 + "/nullvec/null_iter15.words")
cr = parse_counts(REPO + "/vectors/5sub/counts.words.vocab")
cn = parse_counts(SC6 + "/nullvec/counts.words.vocab")

wr = sorted({w for (w, y) in nr if y == 2012 and (w, 2018) in nr})
wn = sorted({w for (w, y) in nn if y == 2012 and (w, 2018) in nn})
print(f"real both-years words: {len(wr):,}   null: {len(wn):,}")

# ------------------------------------------------------------------ GATE
print("\n== GATE: reproduce recorded raw numbers ==")
r12 = np.array([nr[(w, 2012)] for w in wr]); r18 = np.array([nr[(w, 2018)] for w in wr])
n12 = np.array([nn[(w, 2012)] for w in wn]); n18 = np.array([nn[(w, 2018)] for w in wn])
dR, dN = r18 - r12, n18 - n12
totR = {w: sum(cr.get((w, y), 0) for y in YEARS) for w in wr}
totN = {w: sum(cn.get((w, y), 0) for y in YEARS) for w in wn}
lfR = np.log10(np.array([totR[w] for w in wr], float))
lfN = np.log10(np.array([totN[w] for w in wn], float))
print(f"  norm2012 median  real {np.median(r12):.3f} (mem 3.126)   null {np.median(n12):.3f} (mem 3.113)")
print(f"  norm2018 median  real {np.median(r18):.3f} (mem 3.103)   null {np.median(n18):.3f} (mem 3.269)")
print(f"  |dnorm| mean     real {np.abs(dR).mean():.4f} (mem 0.4333)  null {np.abs(dN).mean():.4f} (mem 0.3514)  ratio {np.abs(dN).mean()/np.abs(dR).mean():.3f} (mem 0.811)")
print(f"  dnorm sd         real {dR.std():.4f} (mem 0.6851)  null {dN.std():.4f} (mem 0.4749)  ratio {dN.std()/dR.std():.3f} (mem 0.693)")
print(f"  rho(dnorm, logfreq)  real {spearmanr(dR, lfR).correlation:+.4f} (mem -0.0691)   null {spearmanr(dN, lfN).correlation:+.4f} (mem -0.5719)")

# ------------------------------------------------------------- MODEL f
print("\n== MODEL: f(log10 count) -> norm, fit on null half A ==")
obs_c, obs_n = [], []
for (w, y), c in cn.items():
    if y in (2012, 2013, 2014, 2015, 2016, 2017, 2018) and c > 0 and (w, y) in nn and half(w) == 0:
        obs_c.append(np.log10(c)); obs_n.append(nn[(w, y)])
obs_c, obs_n = np.array(obs_c), np.array(obs_n)
edges = np.arange(obs_c.min(), obs_c.max() + 0.1, 0.1)
bi = np.clip(np.digitize(obs_c, edges) - 1, 0, len(edges) - 2)
centers, means = [], []
for b in range(len(edges) - 1):
    m = bi == b
    if m.sum() >= 50:
        centers.append(edges[b] + 0.05); means.append(obs_n[m].mean())
centers, means = np.array(centers), np.array(means)
f = lambda lc: np.interp(lc, centers, means)  # noqa: E731
print(f"  fit on {len(obs_c):,} (word,year) null obs, {len(centers)} bins, "
      f"norm range {means.min():.2f}..{means.max():.2f} over count 10^{centers.min():.1f}..10^{centers.max():.1f}")
ss_res = float(np.mean((obs_n - f(obs_c)) ** 2)); ss_tot = float(np.var(obs_n))
print(f"  in-sample R^2 of count alone: {1 - ss_res / ss_tot:.3f}")

def excess(words, norms, counts):
    out_w, out_e, out_d, out_p = [], [], [], []
    for w in words:
        c12, c18 = counts.get((w, 2012), 0), counts.get((w, 2018), 0)
        if c12 <= 0 or c18 <= 0:
            continue
        pred = f(np.log10(c18)) - f(np.log10(c12))
        d = norms[(w, 2018)] - norms[(w, 2012)]
        out_w.append(w); out_e.append(d - pred); out_d.append(d); out_p.append(pred)
    return pd.DataFrame({"excess": out_e, "dnorm": out_d, "pred": out_p}, index=out_w)

ER = excess(wr, nr, cr)
EN = excess([w for w in wn if half(w) == 1], nn, cn)          # held-out half B
print(f"  excess computed: real n={len(ER):,}   null held-out n={len(EN):,}")

# ------------------------------------------------------------- FLOOR
print("\n== PRE-REGISTERED FLOOR ==")
sd_ratio = EN.excess.std() / ER.excess.std()
abs_ratio = EN.excess.abs().mean() / ER.excess.abs().mean()
print(f"  mechanical pred median  real {ER.pred.median():+.4f}   null {EN.pred.median():+.4f}")
print(f"  excess median           real {ER.excess.median():+.4f}   null {EN.excess.median():+.4f}")
print(f"  excess sd               real {ER.excess.std():.4f}   null {EN.excess.std():.4f}   RATIO {sd_ratio:.3f}")
print(f"  |excess| mean           real {ER.excess.abs().mean():.4f}   null {EN.excess.abs().mean():.4f}   ratio {abs_ratio:.3f}")
verdict = ("SURVIVES" if sd_ratio <= 0.70 else "DEAD" if sd_ratio >= 0.90 else "INTERMEDIATE")
print(f"  VERDICT (sd ratio, <=0.70 survives / >=0.90 dead): {verdict}")

lf_er = np.array([np.log10(totR[w]) for w in ER.index])
rho_f = spearmanr(ER.excess, lf_er).correlation
print(f"  FREQ check: rho(excess_real, log_freq) = {rho_f:+.4f}   (|rho|<=0.15 = clean)")

# ------------------------------------------------------------- VALIDITY
print("\n== EXTERNAL VALIDITY: norm level vs Brysbaert concreteness ==")
b = pd.read_csv(REPO + "/data/brysbaert_concreteness.txt", sep="\t",
                keep_default_na=False)
conc = {str(w).lower(): c for w, c in zip(b.Word, b["Conc.M"]) if " " not in str(w)}
lvl, lf2, cm = [], [], []
for w in wr:
    if w in conc:
        ns_ = [nr[(w, y)] for y in YEARS if (w, y) in nr]
        if len(ns_) == 7:
            lvl.append(np.mean(ns_)); lf2.append(np.log10(totR[w])); cm.append(conc[w])
lvl, lf2, cm = map(np.asarray, (lvl, lf2, cm))
rl, rf, rc = rankdata(lvl), rankdata(lf2), rankdata(cm)
res = lambda a: a - np.polyval(np.polyfit(rf, a, 1), rf)  # noqa: E731
part = float(np.corrcoef(res(rl), res(rc))[0, 1])
print(f"  n={len(lvl):,}   raw rho(mean norm, conc) = {spearmanr(lvl, cm).correlation:+.4f}"
      f"   freq-partial = {part:+.4f}   (prediction: positive; TN2014 was -0.263/-0.334)")

# ------------------------------------------------------------- EXPLORE
if verdict != "DEAD":
    print("\n== EXPLORATORY (floor not dead) ==")
    sp = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache", verbose=False)
    dw, dv = sp.drift(2012, 2018)
    dr = pd.Series(dict(zip(dw, dv))).reindex(ER.index)
    ok = dr.notna()
    E, D, L = ER.excess[ok].to_numpy(), dr[ok].to_numpy(), lf_er[ok.to_numpy()]
    dec = pd.qcut(L, 10, labels=False, duplicates="drop")
    rhos = [spearmanr(E[dec == i], D[dec == i]).correlation for i in range(10)]
    print(f"  excess x drift  overall {spearmanr(E, D).correlation:+.4f}   "
          f"fm-decile mean {np.mean(rhos):+.4f}")
    print("    per-bin: " + " ".join(f"{r:+.3f}" for r in rhos))
    sen = json.load(open(REPO + "/senses.json"))
    ns = {k: sum(len(x.get("senses", [])) for x in v) for k, v in sen.items()}
    nsv = np.array([ns.get(w, np.nan) for w in ER.index], float)
    nsv[nsv == 0] = np.nan
    m = ~np.isnan(nsv)
    ra, rb_, rz = rankdata(ER.excess[m]), rankdata(nsv[m]), rankdata(lf_er[m])
    resz = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    print(f"  excess x n_senses partial(freq) = "
          f"{float(np.corrcoef(resz(ra), resz(rb_))[0, 1]):+.4f}   n={int(m.sum()):,}")

ER.assign(log_freq=lf_er).to_csv(REPO + "/data/norm_excess/excess_real.csv")
print("\nwrote data/norm_excess/excess_real.csv")
