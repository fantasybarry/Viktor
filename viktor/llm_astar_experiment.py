"""Fair static comparison of ordinary A* vs guided A* (spec Section 7C).

Specification: docs/math/llm_astar_pseudocode.md, Section 7C.

NO LEARNED MODEL EXISTS. The default guide provider, ``straight_line_demo``,
is a SYNTHETIC DEMO: it rounds points on the start-goal segment to lattice
coordinates. It is not Viktor output and results with it say nothing about
learned guidance. A real model plugs in as any ``GuideProvider`` callable.

Protocol per evaluation task (identical graph, costs, ties, collision checks,
and caps for both methods, since both call the same guided_astar):
  * origin = clock() at method start, cutoff = origin + budget_s (same origin
    convention and total budget per method). Guided: guidance time is spent
    before search and so reduces the remaining search time.
  * total time = guidance + filtering + search + rekeys + final validation.
  * valid_by_deadline: GEOMETRIC_CANDIDATE, passes independent validation, and
    validation finished strictly before cutoff (CHECK_TIME convention).
  * Memory pass (separate run, no deadline, same caps): tracemalloc peak of
    Python allocations during guidance+search+validation. Scope: traced Python
    heap only; excludes the pre-built map, interpreter, and any model runtime.
    It is NOT total system memory. Process peak RSS is reported separately.
  * Uncapped reference: ordinary A* without deadline or practical pop cap,
    reported separately from the deadline-limited comparison.
Maps are split as whole maps into train/eval; only eval maps are run.

Usage:  python -m viktor.llm_astar_experiment --help
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Sequence

from viktor.llm_astar import Clock, GridMap, Rect, Status, guided_astar

GuideProvider = Callable[[GridMap, int, int], object]
"""(grid, start ID, goal ID) -> raw guide proposals: list of (x, y) in m."""


def make_map(seed: int, size: int, n_obstacles: int) -> GridMap:
    """Synthetic static map: size x size lattice, random closed boxes (m)."""
    rng = random.Random(seed)
    boxes: list[Rect] = []
    for _ in range(n_obstacles):
        x0, y0 = rng.uniform(0, size - 1), rng.uniform(0, size - 1)
        boxes.append((x0, y0, min(size - 1.0, x0 + rng.uniform(0.2, size / 4)), min(size - 1.0, y0 + rng.uniform(0.2, size / 4))))
    return GridMap(size, size, boxes)


def split_by_map(map_seeds: Sequence[int], eval_fraction: float, rng: random.Random) -> tuple[list[int], list[int]]:
    """Partition whole maps (never individual start/goal pairs) into train/eval."""
    seeds = list(map_seeds)
    rng.shuffle(seeds)
    n_eval = max(1, round(eval_fraction * len(seeds)))
    return sorted(seeds[n_eval:]), sorted(seeds[:n_eval])


def straight_line_demo(k: int) -> GuideProvider:
    """SYNTHETIC DEMO provider (not learned): k rounded points on segment s-goal."""

    def provide(grid: GridMap, s: int, goal: int) -> object:
        (sx, sy), (gx, gy) = grid.position(s), grid.position(goal)
        return [(float(round(sx + (gx - sx) * i / (k + 1))), float(round(sy + (gy - sy) * i / (k + 1)))) for i in range(1, k + 1)]

    return provide


def validate_path(grid: GridMap, path: Sequence[int] | None, s: int, goal: int) -> float | None:
    """Independent check of a returned candidate; returns recomputed length (m).

    Deliberately does not reuse GridMap.valid_edge: each step must be a unit
    axis-aligned move between free in-bounds vertices, and its segment's
    bounding box must not overlap any closed obstacle box (exact for
    axis-aligned segments). Returns None if any check fails.
    """
    if not path or path[0] != s or path[-1] != goal:
        return None
    pts = []
    for v in path:
        if not isinstance(v, int) or not 0 <= v < grid.width * grid.height:
            return None
        x, y = grid.position(v)
        if any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in grid.obstacles):
            return None
        pts.append((x, y))
    length = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        if abs(ax - bx) + abs(ay - by) != 1.0:
            return None
        lo_x, hi_x, lo_y, hi_y = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
        if any(r[0] <= hi_x and lo_x <= r[2] and r[1] <= hi_y and lo_y <= r[3] for r in grid.obstacles):
            return None
        length += math.dist((ax, ay), (bx, by))
    return length


def run_method(
    grid: GridMap,
    s: int,
    goal: int,
    provider: GuideProvider | None,
    budget_s: float,
    p_max: int,
    k_max: int,
    v_max: int,
    clock: Clock,
) -> dict:
    """One method on one task under the shared total budget. provider=None: ordinary A*."""
    origin = clock()
    cutoff = origin + budget_s
    raw, guidance_failure = [], None
    if provider is not None:
        try:
            raw = provider(grid, s, goal)
        except Exception as e:  # optional guidance failed: record it, search with guidance disabled
            raw, guidance_failure = None, type(e).__name__
    t_guided = clock()
    res = guided_astar(grid, s, goal, raw, cutoff, p_max, k_max=k_max, v_max=v_max, clock=clock)
    t_searched = clock()
    length = validate_path(grid, res.path, s, goal) if res.status is Status.GEOMETRIC_CANDIDATE else None
    t_end = clock()
    return {
        "status": res.status.value,
        "guidance_failure": guidance_failure,
        "validated": length is not None,
        "valid_by_deadline": length is not None and t_end < cutoff,
        "length_m": length,
        "pops": res.pops,
        "rekeys": res.rekeys,
        "reopens": res.reopens,
        "guidance_s": t_guided - origin,
        "search_s": t_searched - t_guided,
        "validation_s": t_end - t_searched,
        "total_s": t_end - origin,
    }


def traced_peak_bytes(fn: Callable[[], object]) -> int:
    """tracemalloc peak (bytes) of Python allocations made while fn runs."""
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def run_experiment(
    n_maps: int,
    pairs_per_map: int,
    size: int,
    n_obstacles: int,
    budget_s: float,
    p_max: int,
    k_max: int,
    v_max: int,
    seed: int,
    provider: GuideProvider,
    clock: Clock = time.perf_counter,
) -> dict:
    """Run the Section 7C protocol; returns per-task records plus the split."""
    rng = random.Random(seed)
    train, held_out = split_by_map([seed * 100003 + i for i in range(n_maps)], 0.5, rng)
    records = []
    for map_seed in held_out:
        grid = make_map(map_seed, size, n_obstacles)
        free = sorted(grid.free)
        pair_rng = random.Random(map_seed)
        for pair in range(pairs_per_map):
            if len(free) < 2:
                break
            s, goal = pair_rng.sample(free, 2)
            rec = {"map_seed": map_seed, "pair": pair, "start": grid.position(s), "goal": grid.position(goal)}
            for name, prov in (("baseline", None), ("guided", provider)):
                r = run_method(grid, s, goal, prov, budget_s, p_max, k_max, v_max, clock)

                def full_run(prov: GuideProvider | None = prov) -> None:
                    run_method(grid, s, goal, prov, math.inf, p_max, k_max, v_max, clock)

                r["traced_peak_bytes_uncapped_time"] = traced_peak_bytes(full_run)
                rec[name] = r
            # Uncapped reference: no deadline; pop cap far above |V| (consistent h never reopens).
            ref = guided_astar(grid, s, goal, [], math.inf, 10 * len(grid.free) + 1, k_max=k_max, v_max=v_max, clock=clock)
            rec["reference"] = {"status": ref.status.value, "length_m": validate_path(grid, ref.path, s, goal)}
            records.append(rec)
    return {"train_maps": train, "eval_maps": held_out, "records": records}


def summarize(result: dict) -> dict:
    """Aggregate over ALL admitted tasks; failures and slow cases are kept."""
    recs = result["records"]
    out: dict = {"tasks": len(recs), "eval_maps": len(result["eval_maps"]), "train_maps_unused": len(result["train_maps"])}
    for m in ("baseline", "guided"):
        rs = [r[m] for r in recs]
        ratios = [
            r[m]["length_m"] / r["reference"]["length_m"]
            for r in recs
            if r[m]["validated"] and r["reference"]["length_m"]
        ]
        out[m] = {
            "status_counts": dict(Counter(x["status"] for x in rs)),
            "guidance_failures": sum(x["guidance_failure"] is not None for x in rs),
            "valid_by_deadline_coverage": sum(x["valid_by_deadline"] for x in rs) / len(rs) if rs else None,
            "mean_pops": statistics.fmean(x["pops"] for x in rs) if rs else None,
            "mean_total_ms": 1e3 * statistics.fmean(x["total_s"] for x in rs) if rs else None,
            "max_total_ms": 1e3 * max(x["total_s"] for x in rs) if rs else None,
            "mean_guidance_ms": 1e3 * statistics.fmean(x["guidance_s"] for x in rs) if rs else None,
            "mean_length_ratio_vs_reference": statistics.fmean(ratios) if ratios else None,
            "max_traced_peak_bytes_uncapped_time": max(x["traced_peak_bytes_uncapped_time"] for x in rs) if rs else None,
        }
    out["reference_uncapped"] = {"status_counts": dict(Counter(r["reference"]["status"] for r in recs))}
    out["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB, whole process
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baseline vs guided A* on synthetic static maps (SYNTHETIC DEMO guides; no learned model).")
    ap.add_argument("--maps", type=int, default=10, help="total maps; half (by map) are held out and evaluated")
    ap.add_argument("--pairs", type=int, default=5, help="start/goal pairs per evaluated map")
    ap.add_argument("--size", type=int, default=30, help="lattice side length (vertices, 1 m spacing)")
    ap.add_argument("--obstacles", type=int, default=12)
    ap.add_argument("--budget-ms", type=float, default=20.0, help="total per-method budget from the method's start")
    ap.add_argument("--p-max", type=int, default=5000)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--v-max", type=int, default=100000)
    ap.add_argument("--demo-guides", type=int, default=3, help="points from the synthetic straight-line demo provider")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--records", help="optional path to write all per-task records as JSON")
    a = ap.parse_args(argv)
    result = run_experiment(
        a.maps, a.pairs, a.size, a.obstacles, a.budget_ms / 1e3, a.p_max, a.k_max, a.v_max, a.seed, straight_line_demo(a.demo_guides)
    )
    if a.records:
        with open(a.records, "w") as fh:
            json.dump(result, fh, indent=1)
    summary = summarize(result)
    summary["guide_provider"] = "SYNTHETIC_DEMO straight_line_demo (not a learned model)"
    summary["timing_scope"] = "CPython wall time on this machine; not Thor/vehicle timing, no 20-30 ms guarantee"
    json.dump(summary, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
