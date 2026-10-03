"""Is the evenness-filter gain just frequency selection?

Even year-profiles are a multinomial-stability property: frequent words have
proportional profiles almost automatically. `excess` already rises with
frequency (+0.033 rare -> +0.121 frequent). So the msr filter could be buying
its gain entirely by selecting frequent words. Test within frequency strata.
"""
import re, sys, collections
import numpy as np, pandas as pd

REPO = "/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift"
SC6 = ("/private/tmp/claude-501/-Users-felixstuart-Documents-School-Clubs-"
       "Programming-Club-SemanticDrift/6c0853ee-4ee7-440b-bfeb-04a2b420311a/scratchpad")
YEARS = list(range(2012, 2019)); PAT = re.compile(r"^(.*)_(\d{4})$")


def prof(p):
    per = collections.defaultdict(dict)
    for line in open(p):
        a = line.split()
        if len(a) == 2 and PAT.match(a[0]):
            m = PAT.match(a[0]); per[m.group(1)][int(m.group(2))] = int(a[1])
    ws = [w for w, d in per.items() if len(d) == 7]
    M = np.array([[per[w][y] for y in YEARS] for w in ws], float)
    sh = M.sum(0) / M.sum(); P = (M / M.sum(1, keepdims=True)) / sh
    return pd.DataFrame({"msr": P.min(1) / P.max(1)}, index=ws)


TR = pd.read_csv(REPO + "/data/step_profile/orderperm_real.csv", index_col=0,
                 keep_default_na=False).join(prof(REPO + "/vectors/5sub/counts.words.vocab"),
                                             how="inner")
TN = pd.read_csv(REPO + "/data/step_profile/orderperm_shuffled.csv", index_col=0,
                 keep_default_na=False).join(prof(SC6 + "/nullvec/counts.words.vocab"),
                                             how="inner")
print("median log_freq by evenness bar (real):")
for bar in (0.0, 0.5, 0.7, 0.8, 0.9):
    s = TR[TR.msr >= bar]
    print(f"  msr>={bar:.1f}  n={len(s):>6,}  median log_freq {s.log_freq.median():.2f}"
          f"  (unfiltered {TR.log_freq.median():.2f})")

# common decile edges from the REAL unfiltered distribution, applied to both arms
edges = np.quantile(TR.log_freq, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
for T in (TR, TN):
    T["dec"] = np.digitize(T.log_freq, edges[1:-1])

print("\nWITHIN FREQUENCY DECILE: median excess, msr>=0.7 vs msr<0.7 (real), and shuffled")
print(f"{'dec':>4}{'n even':>8}{'even':>10}{'n uneven':>10}{'uneven':>10}{'diff':>9}"
      f"{'shuf even':>11}{'ratio':>7}")
rows = []
for d in range(10):
    a = TR[(TR.dec == d) & (TR.msr >= 0.7)]; b = TR[(TR.dec == d) & (TR.msr < 0.7)]
    c = TN[(TN.dec == d) & (TN.msr >= 0.7)]
    if len(a) < 30 or len(b) < 30:
        print(f"{d:>4}{len(a):>8,}{'-':>10}{len(b):>10,}{'-':>10}{'-':>9}{'-':>11}{'-':>7}")
        continue
    ma, mb, mc = a.excess.median(), b.excess.median(), c.excess.median()
    rows.append((d, ma, mb, mc))
    print(f"{d:>4}{len(a):>8,}{ma:>+10.4f}{len(b):>10,}{mb:>+10.4f}{ma-mb:>+9.4f}"
          f"{mc:>+11.4f}{mc/ma:>7.2f}")

if rows:
    D = np.array([r[1] - r[2] for r in rows])
    R = np.array([r[3] / r[1] for r in rows])
    print(f"\n  even-minus-uneven within decile: median {np.median(D):+.4f}, "
          f"positive in {int((D > 0).sum())}/{len(D)} deciles")
    print(f"  shuffled/real ratio within decile (even words): median {np.median(R):.2f}, "
          f"max {R.max():.2f}")
    print("\n  => the evenness gain is " +
          ("REAL (survives frequency stratification)" if (D > 0).sum() >= len(D) - 1
           else "FREQUENCY SELECTION (vanishes within decile)"))
