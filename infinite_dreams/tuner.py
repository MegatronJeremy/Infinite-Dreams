# search.py
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from infinite_dreams import infinite_dreams_ext
from infinite_dreams.sizes import default_sizes
from infinite_dreams.utils import set_deterministic, time_cuda_ms, write_csv

# ---------------------------------------------------------------------
# Columns: tile_m, tile_n, tile_k, tm, tn
# ---------------------------------------------------------------------
_CFG_TABLE_T = infinite_dreams_ext.gemm_cfg_table_cpu()
_CFG_TABLE = _CFG_TABLE_T.numpy()
_NUM_CFG = int(_CFG_TABLE.shape[0])

_TUPLE_TO_ID: Dict[Tuple[int, ...], int] = {
    tuple(map(int, row)): int(i) for i, row in enumerate(_CFG_TABLE)
}


@dataclass(frozen=True)
class TileConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    tm: int
    tn: int

    def as_tuple(self) -> Tuple[int, int, int, int, int]:
        return self.tile_m, self.tile_n, self.tile_k, self.tm, self.tn


def cfg_to_id(cfg: TileConfig) -> int:
    return _TUPLE_TO_ID.get(cfg.as_tuple(), -1)


def is_valid_tile(cfg: TileConfig) -> bool:
    return cfg.as_tuple() in _TUPLE_TO_ID


import heapq
from dataclasses import dataclass
from typing import Tuple, Optional, Set


def _score(orig: Tuple[int, int, int, int, int], cand: Tuple[int, int, int, int, int]) -> Tuple[int, int, int]:
    om, on, ok, otm, otn = orig
    m, n, k, tm, tn = cand

    # Weighted distance (tune weights)
    # Usually: changing k is "less harmful" than shrinking m/n a lot,
    # but shrinking k too much can hurt too—so moderate weight.
    dm = abs(m - om) // 8
    dn = abs(n - on) // 8
    dk = abs(k - ok) // 8

    dist = 3 * (dm + dn) + 2 * dk

    # Prefer larger tile area (keep throughput potential)
    # Using negative because heapq is min-heap.
    area_penalty = -(m * n)

    # Prefer not changing tm/tn (stability)
    warp_penalty = (tm != otm) + (tn != otn)

    return dist, warp_penalty, area_penalty


def project_to_valid(cfg: TileConfig,
                     bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]):
    m, n, k, tm, tn = cfg.as_tuple()
    (m_lo, m_hi), (n_lo, n_hi), (k_lo, k_hi) = bounds
    orig = (m, n, k, tm, tn)

    cfg0 = TileConfig(m, n, k, tm, tn)
    if cfg_to_id(cfg0) >= 0:
        return cfg0

    pq = []
    seen: Set[Tuple[int, int, int, int, int]] = set()

    def push(state: Tuple[int, int, int, int, int]):
        if state in seen:
            return
        seen.add(state)
        heapq.heappush(pq, (_score(orig, state), state))

    push(orig)

    while pq:
        _, current_cfg = heapq.heappop(pq)
        m, n, k, tm, tn = current_cfg

        cand = TileConfig(m, n, k, tm, tn)
        if cfg_to_id(cand) >= 0:
            return cand

        if k > k_lo: push((m, n, k - 8, tm, tn))
        if m > m_lo: push((m - 8, n, k, tm, tn))
        if n > n_lo: push((m, n - 8, k, tm, tn))

        if k < k_hi: push((m, n, k + 8, tm, tn))
        if m < m_hi: push((m + 8, n, k, tm, tn))
        if n < n_hi: push((m, n + 8, k, tm, tn))

        if tm < 8: push((m, n, k, tm * 2, tn))
        if tn < 8: push((m, n, k, tm, tn * 2))

        if tm > 1: push((m, n, k, tm // 2, tn))
        if tn > 1: push((m, n, k, tm, tn // 2))

    print("Error: Failed to convert to valid config within budget")
    return cfg


@dataclass(frozen=True)
class BenchCase:
    M: int
    N: int
    K: int
    A: torch.Tensor
    B: torch.Tensor


def make_bench_cases(
        sizes: Sequence[Tuple[int, int, int]],
        seed: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
) -> List[BenchCase]:
    """
    Pre-allocate inputs for benchmarking.
    """
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    cases: List[BenchCase] = []
    for (M, N, K) in sizes:
        A = torch.randn((M, K), device=device, dtype=dtype, generator=g)
        B = torch.randn((K, N), device=device, dtype=dtype, generator=g)
        cases.append(BenchCase(M=M, N=N, K=K, A=A, B=B))
    return cases


def quantize8_int(x: float, lo: int, hi: int) -> int:
    v = int(round(x / 8.0) * 8)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def quantize_pow2_choice(x: float, choices: Tuple[int, ...] = (1, 2, 4, 8)) -> int:
    best = choices[0]
    best_d = abs(float(x) - best)
    for c in choices[1:]:
        d = abs(float(x) - c)
        if d < best_d or (d == best_d and c < best):
            best = c
            best_d = d
    return int(best)


@dataclass
class ScoreOptions:
    iters: int
    warmup: int


class TileScorer:
    """
    Benchmarks configs on preallocated inputs and caches results per discrete 5D tuple.
    Optionally logs EVERY evaluation to a CSV for later visualization.
    """

    def __init__(
            self,
            cases: Sequence[BenchCase],
            score_opts: ScoreOptions,
            eval_log_path: Optional[str] = None,
    ):
        self.cases = list(cases)
        self.opts = score_opts
        self.cache: Dict[Tuple[int, int, int, int, int], float] = {}

        # Eval logging
        self.eval_log_path = eval_log_path
        self._eval_id = 0
        self._csv_writer = None
        self._csv_file = None

        # Context (set by optimizer/confirm loop)
        self.phase = "unknown"
        self.gen = -1
        self.pop_i = -1

        if self.eval_log_path is not None:
            os.makedirs(os.path.dirname(self.eval_log_path) or ".", exist_ok=True)
            new_file = not os.path.exists(self.eval_log_path)
            self._csv_file = open(self.eval_log_path, "a", newline="")
            fieldnames = [
                "eval_id",
                "timestamp_ms",
                "phase",
                "gen",
                "pop_i",
                "cfg_id",
                "tile_m",
                "tile_n",
                "tile_k",
                "tm",
                "tn",
                "ms",
                "cache_hit",
                "valid_cfg",
            ]
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            if new_file:
                self._csv_writer.writeheader()
                self._csv_file.flush()

    def set_context(self, phase: str, gen: int, pop_i: int) -> None:
        self.phase = phase
        self.gen = int(gen)
        self.pop_i = int(pop_i)

    def _log_eval(
            self,
            cfg: TileConfig,
            cfg_id: int,
            ms: float,
            cache_hit: int,
            valid_cfg: int,
    ) -> None:
        if self._csv_writer is None:
            return
        self._csv_writer.writerow(
            dict(
                eval_id=self._eval_id,
                timestamp_ms=int(time.time() * 1000),
                phase=self.phase,
                gen=self.gen,
                pop_i=self.pop_i,
                cfg_id=cfg_id,
                tile_m=cfg.tile_m,
                tile_n=cfg.tile_n,
                tile_k=cfg.tile_k,
                tm=cfg.tm,
                tn=cfg.tn,
                ms=float(ms),
                cache_hit=int(cache_hit),
                valid_cfg=int(valid_cfg),
            )
        )
        self._csv_file.flush()
        self._eval_id += 1

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    @torch.no_grad()
    def score(self, cfg: TileConfig) -> float:
        key = cfg.as_tuple()

        cached = self.cache.get(key, None)
        if cached is not None:
            # log cache-hit too (so you can see the full "path")
            cfg_id = cfg_to_id(cfg)
            valid = 1 if cfg_id >= 0 else 0
            self._log_eval(cfg, cfg_id, cached, cache_hit=1, valid_cfg=valid)
            return cached

        cfg_id = cfg_to_id(cfg)
        if cfg_id < 0:
            ms = 1e9
            self.cache[key] = ms
            self._log_eval(cfg, cfg_id, ms, cache_hit=0, valid_cfg=0)
            return ms

        ms_list: List[float] = []
        for case in self.cases:
            A, B = case.A, case.B

            def run():
                return infinite_dreams_ext.gemm_forward_cfg(A, B, int(cfg_id))

            ms = time_cuda_ms(
                run,
                iters=self.opts.iters,
                warmup=self.opts.warmup,
            )
            ms_list.append(ms)

        score = float(np.mean(ms_list))
        self.cache[key] = score
        self._log_eval(cfg, cfg_id, score, cache_hit=0, valid_cfg=1)
        return score


def differential_evolution_discrete_5d(
        bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        pop_size: int,
        generations: int,
        F: float,
        CR: float,
        scorer: TileScorer,
        verbose: bool = True,
) -> Tuple[TileConfig, float, List[Dict[str, Any]]]:
    rng = np.random.default_rng()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    (m_lo, m_hi), (n_lo, n_hi), (k_lo, k_hi) = bounds

    def rand_vec() -> np.ndarray:
        return np.array(
            [
                rng.uniform(m_lo, m_hi),
                rng.uniform(n_lo, n_hi),
                rng.uniform(k_lo, k_hi),
                rng.uniform(1.0, 8.0),  # tm
                rng.uniform(1.0, 8.0),  # tn
            ],
            dtype=np.float32,
        )

    def vec_to_cfg(vec: np.ndarray) -> TileConfig:
        return project_to_valid(TileConfig(
            quantize8_int(float(vec[0]), m_lo, m_hi),
            quantize8_int(float(vec[1]), n_lo, n_hi),
            quantize8_int(float(vec[2]), k_lo, k_hi),
            quantize_pow2_choice(float(vec[3])),
            quantize_pow2_choice(float(vec[4])),
        ), bounds)

    pop = [rand_vec() for _ in range(pop_size)]
    scores = []
    for i, v in enumerate(pop):
        scorer.set_context("search_init", gen=-1, pop_i=i)
        scores.append(scorer.score(vec_to_cfg(v)))
    scores = np.array(scores, dtype=np.float64)

    best_i = int(np.argmin(scores))
    best_v = pop[best_i].copy()

    history: List[Dict[str, Any]] = []

    for gen in range(generations):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            va, vb, vc = pop[a], pop[b], pop[c]

            mutant = va + F * (vb - vc)

            trial = pop[i].copy()
            jrand = int(rng.integers(0, 5))
            for j in range(5):
                if (rng.random() < CR) or (j == jrand):
                    trial[j] = mutant[j]

            # clamp dims
            trial[0] = np.clip(trial[0], m_lo, m_hi)
            trial[1] = np.clip(trial[1], n_lo, n_hi)
            trial[2] = np.clip(trial[2], k_lo, k_hi)
            trial[3] = np.clip(trial[3], 1.0, 8.0)
            trial[4] = np.clip(trial[4], 1.0, 8.0)

            scorer.set_context("search", gen=gen, pop_i=i)
            s_trial = scorer.score(vec_to_cfg(trial))
            if s_trial < scores[i]:
                pop[i] = trial
                scores[i] = s_trial

        best_i = int(np.argmin(scores))
        best_v = pop[best_i].copy()
        best_s = float(scores[best_i])

        best_cfg = vec_to_cfg(best_v)
        best_id = cfg_to_id(best_cfg)

        history.append(
            dict(
                phase="search",
                gen=gen,
                ms=best_s,
                cfg_id=best_id,
                tile_m=best_cfg.tile_m,
                tile_n=best_cfg.tile_n,
                tile_k=best_cfg.tile_k,
                tm=best_cfg.tm,
                tn=best_cfg.tn,
            )
        )

        if verbose:
            print(f"[gen {gen:03d}] best ms={best_s:.3f} cfg_id={best_id} cfg={best_cfg}")

    final_cfg = vec_to_cfg(best_v)
    final_score = scorer.score(final_cfg)
    return final_cfg, final_score, history


def rerank_topk(
        candidates: List[TileConfig],
        sizes: Sequence[Tuple[int, int, int]],
        iters: int,
        warmup: int,
        seed: Optional[int],
        eval_log_path: Optional[str],
) -> Tuple[List[Tuple[TileConfig, float]], List[Dict[str, Any]]]:
    cases = make_bench_cases(sizes, seed=seed)
    scorer = TileScorer(
        cases,
        ScoreOptions(iters=iters, warmup=warmup),
        eval_log_path=eval_log_path,
    )

    scored: List[Tuple[TileConfig, float]] = []
    rows: List[Dict[str, Any]] = []

    for rank, cfg in enumerate(candidates):
        scorer.set_context("confirm", gen=-1, pop_i=rank)
        ms = scorer.score(cfg)
        scored.append((cfg, ms))
        rows.append(
            dict(
                phase="confirm",
                rank=rank,
                ms=ms,
                cfg_id=cfg_to_id(cfg),
                tile_m=cfg.tile_m,
                tile_n=cfg.tile_n,
                tile_k=cfg.tile_k,
                tm=cfg.tm,
                tn=cfg.tn,
            )
        )

    scored.sort(key=lambda x: x[1])
    return scored, rows


def main():
    ap = argparse.ArgumentParser()

    # Search phase (fast)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--F", type=float, default=0.7)
    ap.add_argument("--CR", type=float, default=0.9)
    ap.add_argument("--search-iters", type=int, default=10)
    ap.add_argument("--search-warmup", type=int, default=2)

    # Confirm phase
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--confirm-iters", type=int, default=100)
    ap.add_argument("--confirm-warmup", type=int, default=20)

    # Output
    ap.add_argument("--out", type=str, default="results/tune_history.csv")
    ap.add_argument("--eval-log", type=str, default="results/eval_log.csv")
    ap.add_argument("--quiet", action="store_true")

    # Bounds for tile dims
    ap.add_argument("--m_lo", type=int, default=8)
    ap.add_argument("--m_hi", type=int, default=256)
    ap.add_argument("--n_lo", type=int, default=8)
    ap.add_argument("--n_hi", type=int, default=256)
    ap.add_argument("--k_lo", type=int, default=8)
    ap.add_argument("--k_hi", type=int, default=256)

    args = ap.parse_args()

    sizes_search = default_sizes()

    bounds = ((args.m_lo, args.m_hi), (args.n_lo, args.n_hi), (args.k_lo, args.k_hi))

    if args.eval_log and os.path.exists(args.eval_log):
        os.remove(args.eval_log)

    # Pre-allocate bench inputs
    cases_search = make_bench_cases(sizes_search, seed=args.seed)
    scorer_search = TileScorer(
        cases_search,
        ScoreOptions(iters=args.search_iters, warmup=args.search_warmup),
        eval_log_path=args.eval_log,
    )

    best_cfg, best_ms, hist = differential_evolution_discrete_5d(
        bounds=bounds,
        pop_size=args.pop,
        generations=args.gen,
        F=args.F,
        CR=args.CR,
        scorer=scorer_search,
        verbose=(not args.quiet),
    )

    cache_items = sorted(scorer_search.cache.items(), key=lambda kv: kv[1])
    top_cfgs: List[TileConfig] = [TileConfig(*k) for (k, _) in cache_items[: args.topk]]

    confirmed, confirm_rows = rerank_topk(
        candidates=top_cfgs,
        sizes=sizes_search,
        iters=args.confirm_iters,
        warmup=args.confirm_warmup,
        seed=args.seed,
        eval_log_path=args.eval_log,
    )

    rows: List[Dict[str, Any]] = []
    rows.extend(hist)
    rows.extend(confirm_rows)

    best_confirm_cfg, best_confirm_ms = confirmed[0]
    rows.append(
        dict(
            phase="best",
            ms=best_confirm_ms,
            cfg_id=cfg_to_id(best_confirm_cfg),
            tile_m=best_confirm_cfg.tile_m,
            tile_n=best_confirm_cfg.tile_n,
            tile_k=best_confirm_cfg.tile_k,
            tm=best_confirm_cfg.tm,
            tn=best_confirm_cfg.tn,
        )
    )

    scorer_search.close()

    write_csv(args.out, rows)

    if not args.quiet:
        print(f"Menu size: {_NUM_CFG} configs")
        print(f"Search-best:   {best_cfg} ms={best_ms:.3f} cfg_id={cfg_to_id(best_cfg)}")
        print(f"Confirm-best:  {best_confirm_cfg} ms={best_confirm_ms:.3f} cfg_id={cfg_to_id(best_confirm_cfg)}")
        print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
