"""Null vs observed for the BleachingTrajectories relationships.

Runs the IDENTICAL battery and frequency-matched test over two spaces:
  - the real 5-sub temporal-referencing vectors, and
  - the time-shuffled null, where pseudo-years carry no temporal information
    so every reported statistic is pure estimation noise.

Anything the real space shows that the null also shows is not semantic change.

usage: compare_null.py <null_vectors> <null_counts_vocab> [--iters N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fm_test import PAIRS, fm_analysis, partial_rho, sp_rho  # noqa: E402
from sense_clusters import sense_features  # noqa: E402

REPO = Path("/Users/felixstuart/Documents/School/Clubs/Programming Club/SemanticDrift")
YEARS = list(range(2012, 2019))
SENSE_THRESHOLD = 0.62   # similarity cut; sensitivity checked at 0.55 / 0.70
SENSE_K = 50             # k=25 is resolution-starved: 10-16% of words score an
                         # exact 0 change, which alone drags sign-agreement across
                         # thresholds from ~86% down to ~56%.


def step_chain(df):
    """Per-word Spearman(step, chain) across that word's year-transitions.

    Negative = the word's larger steps land farther from where its prior
    neighbourhood sat ("detachment"). Vectorised as Pearson on within-word ranks,
    which is exactly Spearman.

    NOTE the mechanical prior: `chain` is the closest any year-t neighbour sits to
    the year-t+1 self vector, and `step` is how far the self vector moved from the
    centre of that same neighbourhood. A larger step therefore lands farther from
    the whole neighbourhood almost by construction, so a strongly negative value is
    expected even with no semantics involved. The null decides whether ANY of it is
    temporal.
    """
    d = df.copy()
    r = d.groupby("word")[["step", "chain"]].rank()
    d["rs"], d["rc"] = r["step"], r["chain"]
    d["p"], d["s2"], d["c2"] = d.rs * d.rc, d.rs ** 2, d.rc ** 2
    g = d.groupby("word")
    n, sx, sy = g.size(), g["rs"].sum(), g["rc"].sum()
    sxy, sxx, syy = g["p"].sum(), g["s2"].sum(), g["c2"].sum()
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    out = ((n * sxy - sx * sy) / den).where(den > 0)
    return out[n >= 4]          # <4 transitions is meaningless


def build_table(vectors_path, counts_path, cache_dir):
    """Full bleaching battery for one space -> per-word table."""
    sys.path.insert(0, str(REPO))
    from driftcache import Space

    sp = Space(str(vectors_path), cache_dir=str(cache_dir), verbose=True)
    agg = sp.agg().copy()
    df = sp.chaining()
    g = df.groupby("word")
    agg["conc_steps"] = g["step"].max() / g["step"].sum()

    counts = sp.counts(str(counts_path))
    total = {w: sum(d.values()) for w, d in counts.items()}
    agg["log_freq"] = np.log10(agg.index.map(total).astype(float))

    words, dist = sp.drift(2012, 2018)
    agg["drift"] = pd.Series(dist, index=words)
    agg["step_chain"] = step_chain(df)

    # word-centred peripheral-graph sense clustering (Kolli et al. 2026, distributional
    # half). Only the DIFFERENCE/TREND features go in; the levels are kept alongside
    # purely as the frequency-confound control they are expected to be.
    # separate k=50 Space used ONLY as a neighbour source -- its own tn/chaining
    # would be computed over 50 neighbours and must not leak into the main battery.
    sp50 = Space(str(vectors_path), cache_dir=str(cache_dir), k=SENSE_K, verbose=True)
    sense, _ = sense_features(sp50, YEARS, k=SENSE_K, verbose=True)
    for col in ("dommass_change", "dommass_slope", "nclust_change",
                "dommass_mean", "nclust_mean"):
        agg[col] = sense[SENSE_THRESHOLD][col]

    agg["tn_first"] = pd.Series(sp.tn[2012])
    agg["tn_mid"] = pd.Series(sp.tn[2015])

    return agg[agg["n_trans"] == len(YEARS) - 1].dropna(
        subset=["tn_first", "tn_mid", "drift", "concentration", "conc_steps", "log_freq"]
    ).copy()


def main():
    null_vecs, null_counts = sys.argv[1], sys.argv[2]
    label = "null(iter?)"
    if "--iters" in sys.argv:
        label = f"null(iter {sys.argv[sys.argv.index('--iters') + 1]})"

    print("=== building REAL 5-sub table ===", flush=True)
    real = build_table(REPO / "vectors/5sub/sgns.words",
                       REPO / "vectors/5sub/counts.words.vocab", REPO / "cache")
    print(f"real: {len(real):,} complete-case words", flush=True)

    print(f"=== building {label} table ===", flush=True)
    null = build_table(null_vecs, null_counts,
                       Path(__file__).parent / "cache_null")
    print(f"null: {len(null):,} complete-case words", flush=True)

    shared = real.index.intersection(null.index)
    print(f"shared vocabulary: {len(shared):,} words", flush=True)

    out = []
    for name, t in (("real", real), (label, null), (f"{label}, shared vocab", null.loc[shared]),
                    ("real, shared vocab", real.loc[shared])):
        res = fm_analysis(t, "log_freq")
        out.append((name, res, len(t)))

    w = 26
    print("\n" + "=" * 100)
    print("NOISE FLOOR: same battery, same code, on real vs time-shuffled data")
    print("=" * 100)
    for label_pair, ca, cb in PAIRS:
        print(f"\n{label_pair}")
        print(f"  {'space':<{w}} {'n':>9} {'overall':>9} {'partial':>9} "
              f"{'fm-bin mean':>12} {'spread':>8} {'sig':>7}")
        for name, res, n in out:
            d = res["pairs"][label_pair]
            print(f"  {name:<{w}} {n:>9,} {d['overall']:>+9.4f} {d['partial']:>+9.4f} "
                  f"{d['bin_mean']:>+12.4f} {d['bin_spread']:>8.3f} "
                  f"{str(d['n_sig']) + '/' + str(len(d['bins'])):>7}")
        for name, res, _ in out:
            d = res["pairs"][label_pair]
            print(f"    {name:<{w}} per fm-bin: "
                  + " ".join(f"{b['rho']:+.3f}" for b in d["bins"]))

    # distributional comparison of the statistics themselves
    print("\n" + "=" * 100)
    print("STATISTIC DISTRIBUTIONS (median [IQR])")
    print("=" * 100)
    print(f"  {'metric':<18} {'real':>28} {'null':>28}")
    for col in ("tn_first", "concentration", "conc_steps", "drift", "mean_ratio",
                "max_ratio", "chain_mean", "chain_rise", "step_chain",
                "dommass_change", "dommass_slope", "nclust_change",
                "dommass_mean", "nclust_mean"):
        def fmt(t):
            if col not in t.columns:
                return "n/a"
            q1, m, q3 = t[col].quantile([.25, .5, .75])
            return f"{m:.4f} [{q1:.4f},{q3:.4f}]"
        print(f"  {col:<18} {fmt(real):>28} {fmt(null):>28}")
    print(f"  {'share max_ratio>=1':<18} {(real['max_ratio'] >= 1).mean():>28.4f} "
          f"{(null['max_ratio'] >= 1).mean():>28.4f}")

    # pre-registered 2026-08-20, BEFORE the null was computed:
    #   null within ~0.05 of the real median  -> step_chain carries no temporal
    #     information and the structural component is gone.
    #   null meaningfully less negative (>= -0.35) -> real data has directional
    #     structure that shuffling destroys; detachment is measurable.
    if "step_chain" in real.columns and "step_chain" in null.columns:
        rm = real["step_chain"].median()
        nm = null["step_chain"].median()
        print("\n" + "=" * 100)
        print("PRE-REGISTERED TEST: step_chain")
        print("=" * 100)
        print(f"  real median   = {rm:+.4f}  (n={real['step_chain'].notna().sum():,}, "
              f"{(real['step_chain'] < 0).mean() * 100:.1f}% negative)")
        print(f"  null median   = {nm:+.4f}  (n={null['step_chain'].notna().sum():,}, "
              f"{(null['step_chain'] < 0).mean() * 100:.1f}% negative)")
        print(f"  gap           = {nm - rm:+.4f}")
        if abs(nm - rm) <= 0.05:
            print("  VERDICT: null matches real within 0.05 -> NO temporal information.")
        elif nm >= -0.35:
            print("  VERDICT: null clearly less negative -> real directional structure survives.")
        else:
            print("  VERDICT: intermediate. Null is more negative than -0.35 but differs from")
            print("           real by more than 0.05 -- report the gap, do not call it either way.")
        print(f"  Spearman(step_chain, log_freq): real "
              f"{sp_rho(real['step_chain'].to_numpy(float), real['log_freq'].to_numpy(float))[0]:+.4f}"
              f"   null "
              f"{sp_rho(null['step_chain'].to_numpy(float), null['log_freq'].to_numpy(float))[0]:+.4f}")

    # ---- PRE-REGISTERED 2026-08-20, before any null was computed ----
    # Under a random-year shuffle every pseudo-year samples the same distribution,
    # so dominant mass should be near-constant across pseudo-years and
    # dommass_change should collapse toward zero. Null |change| meaningfully
    # SMALLER than real => survives. Null |change| matching real => it goes the way
    # of concentration and step_chain.
    print("\n" + "=" * 100)
    print("PRE-REGISTERED TEST: sense-graph change features")
    print("=" * 100)
    print(f"  {'feature':<16} {'real |mean|':>12} {'null |mean|':>12} {'ratio':>8} "
          f"{'real sd':>9} {'null sd':>9} {'sd ratio':>9}")
    for col in ("dommass_change", "dommass_slope", "nclust_change"):
        if col not in real.columns or col not in null.columns:
            continue
        ra, na = real[col].abs().mean(), null[col].abs().mean()
        rs, ns = real[col].std(), null[col].std()
        print(f"  {col:<16} {ra:>12.4f} {na:>12.4f} {na / ra if ra else float('nan'):>8.3f} "
              f"{rs:>9.4f} {ns:>9.4f} {ns / rs if rs else float('nan'):>9.3f}")
    dc_r, dc_n = real.get("dommass_change"), null.get("dommass_change")
    if dc_r is not None and dc_n is not None:
        ratio = dc_n.abs().mean() / dc_r.abs().mean()
        print()
        if ratio <= 0.70:
            print(f"  VERDICT: null change is {ratio:.2f}x real -- SURVIVES, shuffling "
                  f"destroys real structure.")
        elif ratio >= 0.90:
            print(f"  VERDICT: null change is {ratio:.2f}x real -- DEAD, same as "
                  f"concentration/step_chain.")
        else:
            print(f"  VERDICT: null change is {ratio:.2f}x real -- intermediate, "
                  f"report the ratio, do not call it.")
        print(f"  Spearman(dommass_change, log_freq): real "
              f"{sp_rho(dc_r.to_numpy(float), real['log_freq'].to_numpy(float))[0]:+.4f}"
              f"   null "
              f"{sp_rho(dc_n.to_numpy(float), null['log_freq'].to_numpy(float))[0]:+.4f}")

    pd.concat([real.assign(space="real"), null.assign(space="shuffled")]).to_csv(
        Path(__file__).parent / "null_vs_real_table.csv")


if __name__ == "__main__":
    main()
