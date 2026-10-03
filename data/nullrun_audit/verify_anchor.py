"""Anchor-sharing table: sample-dependence + noise-floor check.

Real space full-vocab already computed (+0.275/+0.267...). Now:
  (a) real space restricted to S3's 28,621 words  -> should reproduce memory
  (b) shuffled null space, full vocab             -> what pure noise produces
  (c) shuffled null space, S3-restricted          -> apples-to-apples with (a)
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = "/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad"
SC10 = "/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-Programming-Club-SemanticDrift/10425496-07a9-47a3-8abd-65f4fccd6a77/scratchpad"
sys.path.insert(0, REPO)
from driftcache import Space  # noqa: E402


def partial_rho(x, y, z):
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if m.sum() < 10:
        return np.nan, 0
    rx, ry, rz = (rankdata(v[m]) for v in (x, y, z))
    f = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(f(rx), f(ry))[0, 1]), int(m.sum())


S3 = pd.read_pickle(SC10 + "/S3.pkl")
sen = json.load(open(REPO + "/senses.json"))
ns = {k: sum(len(b.get("senses", [])) for b in v) for k, v in sen.items()}

MEM = {2012: (+0.098, +0.084, +0.045), 2013: (+0.054, +0.088, +0.062),
       2014: (-0.002, +0.035, +0.063), 2015: (-0.083, -0.066, +0.059)}


def table(sp, vocab_path, restrict, label, mem=False):
    counts = sp.counts(vocab_path)
    tot = {w: sum(d.values()) for w, d in counts.items()}
    words = [w for w in sp.words[2012] if w in sp.index[2018]]
    if restrict is not None:
        rs = set(restrict)
        words = [w for w in words if w in rs]
    base = pd.DataFrame(index=words)
    base["log_freq"] = np.log10(pd.Series({w: tot.get(w, np.nan) for w in words}))
    base["tn2012"] = pd.Series(sp.tn[2012]).reindex(words)
    base["tn2014"] = pd.Series(sp.tn[2014]).reindex(words)
    base["n_senses"] = pd.Series({w: ns.get(w, np.nan) for w in words})
    base.loc[base.n_senses == 0, "n_senses"] = np.nan
    print(f"\n  [{label}]  n={len(base):,}")
    hdr = f"  {'anchor':>12} {'tn2012':>20} {'tn2014':>20} {'n_senses':>20}"
    print(hdr)
    for a in (2012, 2013, 2014, 2015):
        ww, vv = sp.drift(a, 2018)
        d = pd.Series(dict(zip(ww, vv))).reindex(base.index)
        cells = []
        for j, col in enumerate(["tn2012", "tn2014", "n_senses"]):
            r, n = partial_rho(base[col].to_numpy(float), d.to_numpy(float),
                               base.log_freq.to_numpy(float))
            s = f"{r:+.3f}"
            if mem:
                s += f" (mem {MEM[a][j]:+.3f})"
            cells.append(s)
        print(f"  {a}->2018   " + " ".join(f"{c:>20}" for c in cells))


sp_real = Space(REPO + "/vectors/5sub/sgns.words", cache_dir=REPO + "/cache",
                verbose=False)
table(sp_real, REPO + "/vectors/5sub/counts.words.vocab", S3.index,
      "REAL, S3-restricted (28,621 sentpos-covered words)", mem=True)

sp_null = Space(SC6 + "/nullvec/null_iter15.words", cache_dir=SC6 + "/cache_null",
                verbose=False)
table(sp_null, SC6 + "/nullvec/counts.words.vocab", None,
      "SHUFFLED NULL, full vocab")
table(sp_null, SC6 + "/nullvec/counts.words.vocab", S3.index,
      "SHUFFLED NULL, S3-restricted")
