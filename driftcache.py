"""Disk-backed cache for the diachronic vector artifacts.

The notebook cold start is dominated by two things: parsing the ~500 MB
word2vec text file, and the all-pairs KNN sweep over every year. Both are pure
functions of the vectors file, so they only ever need computing once.

Everything here is keyed on (path, size, mtime) of the vectors file. Retrain and
the key changes, so the cache rebuilds itself; nothing to remember to delete.

Typical use from a notebook:

    from driftcache import Space
    sp = Space("./vectors/5sub/sgns.words")   # warm: ~1 s, cold: a few minutes

    sp.mat[2012]        # (n_words, 100) float32, L2-normalised, mmapped
    sp.words[2012]      # list[str]
    sp.index[2012]      # {word: row}
    sp.tn[2012]["fam"]  # mean cosine distance to 25 NN
    sp.neighbors(2012, "fam")   # [(word, sim), ...] sorted descending
    sp.chaining()       # tidy DataFrame: word, year, step, radius, ratio, chain
    sp.agg()            # per-word rollup (concentration, chain_rise, ...)

Rebuild a single stage after changing its code with drop=:

    sp = Space("./vectors/5sub/sgns.words", drop=["knn"])
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

YEARS = tuple(range(2012, 2019))
K = 25

# A block of rows scored against the full year at once. 2048 x ~74k float32 is
# ~600 MB of scratch, which keeps the GEMM in cache-friendly territory without
# risking a swap storm on a 16 GB box.
BLOCK = 2048


def _fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class Space:
    def __init__(
        self,
        vectors_path: str | Path,
        cache_dir: str | Path = "cache",
        years: tuple[int, ...] = YEARS,
        k: int = K,
        drop: list[str] | None = None,
        verbose: bool = True,
    ):
        self.vectors_path = Path(vectors_path)
        self.years = tuple(years)
        self.k = k
        self.verbose = verbose
        self.dir = Path(cache_dir) / _fingerprint(self.vectors_path)
        self.dir.mkdir(parents=True, exist_ok=True)

        for stage in drop or []:
            for p in self.dir.glob(f"{stage}*"):
                p.unlink()

        self._load_vectors()
        self._load_knn()
        self._chain_df = None
        self._agg_df = None

    # -- logging ----------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    # -- stage 1: vectors -------------------------------------------------

    def _build_vectors(self):
        """Parse the word2vec text dump into one year-sorted float32 array.

        Rows are grouped by year so each year is a contiguous slice, which lets
        the warm path hand out views of an mmap instead of copying.
        """
        t0 = time.time()
        self._log(f"[driftcache] parsing {self.vectors_path} ...")

        with open(self.vectors_path) as f:
            n_rows, dim = (int(x) for x in f.readline().split())
            keys = np.empty(n_rows, dtype=object)
            key_years = np.empty(n_rows, dtype=np.int32)
            vecs = np.empty((n_rows, dim), dtype=np.float32)
            for i, line in enumerate(f):
                key, rest = line.split(" ", 1)
                base, year = key.rsplit("_", 1)
                keys[i] = base
                key_years[i] = int(year)
                vecs[i] = np.fromstring(rest, dtype=np.float32, sep=" ")

        # Stable sort by year keeps each year's rows in their original file
        # order, so row indices stay comparable with the old notebook code.
        order = np.argsort(key_years, kind="stable")
        vecs = vecs[order]
        keys = keys[order]
        key_years = key_years[order]

        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        bounds = {}
        for y in self.years:
            hit = np.flatnonzero(key_years == y)
            if hit.size == 0:
                raise ValueError(f"no vectors tagged _{y} in {self.vectors_path}")
            bounds[y] = (int(hit[0]), int(hit[-1]) + 1)

        np.save(self.dir / "vectors.npy", vecs)
        (self.dir / "words.json").write_text(json.dumps(list(keys)))
        (self.dir / "bounds.json").write_text(
            json.dumps({str(k): v for k, v in bounds.items()})
        )
        self._log(f"[driftcache] vectors built in {time.time() - t0:.1f}s")

    def _load_vectors(self):
        if not (self.dir / "vectors.npy").exists():
            self._build_vectors()

        self._vecs = np.load(self.dir / "vectors.npy", mmap_mode="r")
        all_words = json.loads((self.dir / "words.json").read_text())
        bounds = {
            int(k): v
            for k, v in json.loads((self.dir / "bounds.json").read_text()).items()
        }

        self.mat, self.words, self.index = {}, {}, {}
        for y in self.years:
            lo, hi = bounds[y]
            self.mat[y] = self._vecs[lo:hi]
            self.words[y] = all_words[lo:hi]
            self.index[y] = {w: i for i, w in enumerate(self.words[y])}

    # -- stage 2: knn + tn ------------------------------------------------

    def _build_knn(self, year: int):
        """Top-k neighbours for every word in `year`, by blocked GEMM.

        The per-word `M @ vec` loop this replaces re-streams the whole year
        matrix once per word; batching turns that into a handful of BLAS calls.
        """
        t0 = time.time()
        M = np.ascontiguousarray(self.mat[year])
        n = M.shape[0]
        idx = np.empty((n, self.k), dtype=np.int32)
        sim = np.empty((n, self.k), dtype=np.float32)

        for lo in range(0, n, BLOCK):
            hi = min(lo + BLOCK, n)
            sims = M[lo:hi] @ M.T
            # Blank out self-similarity so it can't occupy a neighbour slot.
            sims[np.arange(hi - lo), np.arange(lo, hi)] = -np.inf
            part = np.argpartition(sims, -self.k, axis=1)[:, -self.k :]
            vals = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-vals, axis=1)
            idx[lo:hi] = np.take_along_axis(part, order, axis=1)
            sim[lo:hi] = np.take_along_axis(vals, order, axis=1)

        np.savez(self._knn_path(year), idx=idx, sim=sim)
        self._log(f"[driftcache] knn {year} ({n} words) in {time.time() - t0:.1f}s")

    def _knn_path(self, year: int) -> Path:
        """k=25 keeps the legacy filename so existing caches stay valid; any other
        k gets its own file. Without this the loader returns whatever k happens to
        be on disk, silently ignoring the k that was asked for."""
        return self.dir / (
            f"knn_{year}.npz" if self.k == 25 else f"knn_{year}_k{self.k}.npz"
        )

    def _load_knn(self):
        self.knn_idx, self.knn_sim, self.tn = {}, {}, {}
        for y in self.years:
            f = self._knn_path(y)
            if not f.exists():
                self._build_knn(y)
            z = np.load(f)
            self.knn_idx[y] = z["idx"]
            self.knn_sim[y] = z["sim"]
            # TN = mean cosine DISTANCE to the k nearest neighbours
            # (high = diffuse, low = tight).
            dist = 1.0 - self.knn_sim[y].mean(axis=1)
            self.tn[y] = dict(zip(self.words[y], dist.tolist()))

    def neighbors(self, year: int, word: str) -> list[tuple[str, float]]:
        i = self.index[year][word]
        ws = self.words[year]
        return [
            (ws[j], float(s))
            for j, s in zip(self.knn_idx[year][i], self.knn_sim[year][i])
        ]

    def drift(self, y1: int, y2: int, metric: str = "distance"):
        """(words, drift) for every word present in both years.

        metric="distance" (DEFAULT): 1 - cos, so HIGHER = MORE CHANGE.
            Sorting/binning ascending puts the most STABLE words first.
        metric="cosine": raw cosine similarity, HIGHER = MORE STABLE.
            Sorting/binning ascending puts the most CHANGED words first.

        The old behaviour was bare cosine under the name `drift`, which reads
        backwards -- bin 0 was the most-changed words, not the least. Default
        is now distance so the name and the ordering agree.

        Vectors are already L2-normalised, so the dot product is the cosine.
        """
        if metric not in ("distance", "cosine"):
            raise ValueError(f"metric must be 'distance' or 'cosine', got {metric!r}")
        idx1, idx2 = self.index[y1], self.index[y2]
        words = [w for w in self.words[y1] if w in idx2]
        a = np.asarray(self.mat[y1][[idx1[w] for w in words]])
        b = np.asarray(self.mat[y2][[idx2[w] for w in words]])
        cos = np.einsum("nd,nd->n", a, b)
        return words, (1.0 - cos) if metric == "distance" else cos

    # -- stage 3: chaining ------------------------------------------------

    def chaining(self):
        """Per (word, year) step / radius / ratio / chain, as a tidy frame."""
        import pandas as pd

        if self._chain_df is not None:
            return self._chain_df

        f = self.dir / "chaining.parquet"
        if f.exists():
            self._chain_df = pd.read_parquet(f)
            return self._chain_df

        t0 = time.time()
        frames = []
        for year in self.years[:-1]:
            Ma = np.ascontiguousarray(self.mat[year])
            Mb = np.ascontiguousarray(self.mat[year + 1])
            idx_b = self.index[year + 1]

            words = self.words[year]
            keep = np.array(
                [i for i, w in enumerate(words) if w in idx_b], dtype=np.int64
            )
            rows_b = np.array([idx_b[words[i]] for i in keep], dtype=np.int64)

            va = Ma[keep]
            vb = Mb[rows_b]
            step = 1.0 - np.einsum("nd,nd->n", va, vb)

            nbr = self.knn_idx[year][keep]  # (n, k) rows into Ma
            chain = np.empty(keep.size, dtype=np.float32)
            for lo in range(0, keep.size, BLOCK):
                hi = min(lo + BLOCK, keep.size)
                # (b, k, d) neighbour vectors dotted with each word's next-year
                # vector -> the closest any 2012 neighbour sits to the 2013 self.
                chain[lo:hi] = np.einsum("nkd,nd->nk", Ma[nbr[lo:hi]], vb[lo:hi]).max(
                    axis=1
                )

            radius = np.array([self.tn[year][words[i]] for i in keep], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(radius > 1e-9, step / radius, np.nan)

            frames.append(
                pd.DataFrame(
                    {
                        "word": [words[i] for i in keep],
                        "year": year,
                        "step": step,
                        "radius": radius,
                        "ratio": ratio,
                        "chain": chain,
                    }
                )
            )

        df = pd.concat(frames, ignore_index=True).sort_values(
            ["word", "year"], ignore_index=True
        )
        df.to_parquet(f, index=False)
        self._log(f"[driftcache] chaining built in {time.time() - t0:.1f}s")
        self._chain_df = df
        return df

    def agg(self):
        """Per-word rollup of the chaining frame (concentration, chain_rise, ...)."""
        import pandas as pd

        if self._agg_df is not None:
            return self._agg_df

        df = self.chaining()
        g = df.groupby("word")
        agg = pd.DataFrame(
            {
                "n_trans": g["ratio"].size(),
                "mean_ratio": g["ratio"].mean(),
                "max_ratio": g["ratio"].max(),
                "chain_first": g["chain"].first(),
                "chain_last": g["chain"].last(),
                "chain_mean": g["chain"].mean(),
            }
        )
        agg["concentration"] = agg["max_ratio"] / g["ratio"].sum()
        agg["chain_rise"] = (
            agg["chain_last"] - agg["chain_first"]
        )  # + = extension, - = detachment

        peak = df.loc[g["ratio"].idxmax()].set_index("word")
        agg["peak_year"] = peak["year"]
        agg["chain_at_peak"] = peak["chain"]
        agg["chain_drop"] = agg["chain_mean"] - agg["chain_at_peak"]

        self._agg_df = agg
        return agg

    # -- counts (cheap, but cached for symmetry) --------------------------

    def counts(self, vocab_path: str | Path):
        """{word: {year: count}} from a counts.words.vocab file."""
        from collections import defaultdict

        counts = defaultdict(dict)
        with open(vocab_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                token, cnt = parts
                if "_" not in token:
                    continue
                word, year = token.rsplit("_", 1)
                try:
                    counts[word][int(year)] = int(cnt)
                except ValueError:
                    continue
        return counts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Warm the driftcache for a vectors file.")
    ap.add_argument("vectors", nargs="?", default="./vectors/5sub/sgns.words")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument(
        "--drop",
        nargs="*",
        default=None,
        help="stages to rebuild: vectors, words, bounds, knn, chaining",
    )
    args = ap.parse_args()

    t0 = time.time()
    sp = Space(args.vectors, cache_dir=args.cache_dir, drop=args.drop)
    sp.chaining()
    sp.agg()
    peak = 0
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    except ImportError:
        pass
    print(f"[driftcache] total {time.time() - t0:.1f}s, peak RSS {peak:.1f} GB")
    print(f"[driftcache] cache at {sp.dir}")
