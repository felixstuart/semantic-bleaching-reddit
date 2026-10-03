"""Null reference for the exploratory excess-norm x drift association.
Same f (fit on null half A), excess on held-out half B, correlated against the
null space's own pseudo-drift(2012,2018), fm-decile-binned by log_freq.
Result 2026-08-24: overall -0.0937, fm-decile mean -0.1269 (real: -0.1041 /
-0.0530) -> null exceeds real, association is shared estimation noise. DEAD.
Run from repo root; needs the null vectors + cache from the old scratchpad
(originals on RICK work-null if /private/tmp was purged).
"""
# (executable version was run as a heredoc; see results.txt for output and
#  norm_excess.py for the shared parsing/f-fitting code this reuses)
