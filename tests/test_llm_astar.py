"""Tests for viktor.llm_astar and viktor.llm_astar_experiment.

Spec: docs/math/llm_astar_pseudocode.md. Run from the repo root:
    python -m unittest discover -s tests -v
"""

import math
import random
import sys
import unittest

from viktor.llm_astar import (
    DeadlineExceeded,
    GridMap,
    InvalidPath,
    Status,
    _IndexedHeap,
    filter_guides,
    guided_astar,
    reconstruct,
)
from viktor.llm_astar_experiment import run_method, split_by_map, validate_path

INF = math.inf
ZERO = lambda: 0.0  # noqa: E731  clock that never expires
CAPS = dict(k_max=8, v_max=1000)


def run(grid, s, goal, guides, p_max=100, cutoff=INF, clock=ZERO, **caps):
    """Search with (x, y) start/goal; returns result and path as positions."""
    kw = {**CAPS, **caps}
    r = guided_astar(grid, grid.vertex_at(*s), grid.vertex_at(*goal), guides, cutoff, p_max, clock=clock, **kw)
    return r, (None if r.path is None else [grid.position(v) for v in r.path])


def thin_wall(x0, y0, x1, y1):
    """Closed box containing no lattice point when it lies strictly between lattice lines."""
    return (x0, y0, x1, y1)


class RecordingGrid(GridMap):
    """GridMap that records expansion order (successors() is called once per expansion)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.expanded = []

    def successors(self, u):
        self.expanded.append(u)
        return super().successors(u)


def oracle(grid, s, goal, guide_ids, p_max, stale=False):
    """Independent naive reference of Section 7A semantics.

    OPEN is a dict scanned linearly; with stale=False every pop recomputes keys
    from the current g and Q (so keys are never stale). stale=True keeps the
    key from insertion time, i.e. omits REKEY_ALL, to prove tests discriminate.
    Returns (status, cost, pops, expansion order, reopens).
    """
    q, g, closed, exp, pops, reopens = list(guide_ids), {s: 0.0}, set(), [], 0, 0
    gp = grid.position(goal)

    def key(v):
        p = grid.position(v)
        h = math.dist(p, gp)
        return (g[v] + h + (math.dist(p, grid.position(q[0])) if q else 0.0), h, v)

    open_ = {s: key(s)}
    while open_:
        if pops >= p_max:
            return ("POP_LIMIT_REACHED", None, pops, exp, reopens)
        u = min(open_, key=(lambda v: open_[v]) if stale else key)
        del open_[u]
        pops += 1
        if u == goal:
            return ("GEOMETRIC_CANDIDATE", g[goal], pops, exp, reopens)
        closed.add(u)
        exp.append(u)
        while q and q[0] in closed:
            q.pop(0)
        for v in GridMap.successors(grid, u):
            if not grid.valid_edge(u, v, ZERO, INF):
                continue
            c = g[u] + grid.cost(u, v)
            if c < g.get(v, INF):
                g[v] = c
                if v in closed:
                    closed.discard(v)
                    reopens += 1
                open_[v] = key(v)
    return ("NO_PATH_IN_GRAPH", None, pops, exp, reopens)


class TripClock:
    """Deterministic clock: 0.0 until the nth call made from function ``where``
    (via check_time or directly), then ``value`` forever (monotonic)."""

    def __init__(self, where, value, nth=1):
        self.where, self.value, self.nth, self.calls, self.tripped = where, value, nth, 0, False

    def __call__(self):
        if not self.tripped:
            f = sys._getframe(1)
            if f.f_code.co_name == "check_time":
                f = f.f_back
            if f.f_code.co_name == self.where:
                self.calls += 1
                self.tripped = self.calls >= self.nth
        return self.value if self.tripped else 0.0


class SpecExamples(unittest.TestCase):
    """Spec Sections 8-9 on the obstacle-free 3x2 grid, start (0,0), goal (2,0)."""

    G = GridMap(3, 2)
    PATH = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]

    def test_guide_on_path(self):
        r, path = run(self.G, (0, 0), (2, 0), [(1, 0)], p_max=3)
        self.assertEqual((r.status, path, r.pops), (Status.GEOMETRIC_CANDIDATE, self.PATH, 3))
        self.assertEqual(r.cost, 2.0)  # sum of exact 1 m edges is exact
        self.assertEqual(r.rekeys, 1)  # expanding (1,0) exhausts Q -> rekey to b = 0

    def test_guide_off_path_not_visited(self):
        r, path = run(self.G, (0, 0), (2, 0), [(1, 1)], p_max=3)
        self.assertEqual((r.status, path, r.pops, r.cost), (Status.GEOMETRIC_CANDIDATE, self.PATH, 3, 2.0))
        self.assertNotIn(self.G.vertex_at(1, 1), r.path)

    def test_tie_value(self):
        # Goal key f = 2 + 0 + sqrt2, guide key f = 2 + sqrt2 + 0: exactly equal in binary64,
        # so the h field (0 < sqrt2) decides. Spec tolerance 1e-12 on the value.
        f_goal = 2.0 + 0.0 + math.dist((2, 0), (1, 1))
        f_guide = 2.0 + math.dist((1, 1), (2, 0)) + 0.0
        self.assertEqual(f_goal, f_guide)
        self.assertAlmostEqual(f_goal, 3.414213562373095, delta=1e-12)

    def test_pop_limit(self):
        for guides in ([(1, 0)], [(1, 1)], []):
            r, path = run(self.G, (0, 0), (2, 0), guides, p_max=2)
            self.assertEqual((r.status, path, r.pops), (Status.POP_LIMIT_REACHED, None, 2))

    def test_start_equals_goal(self):
        r, path = run(self.G, (1, 1), (1, 1), [(0, 0)], p_max=1)
        self.assertEqual((r.status, path, r.cost, r.pops), (Status.GEOMETRIC_CANDIDATE, [(1.0, 1.0)], 0.0, 1))

    def test_filter_example(self):
        q = filter_guides([(0, 0), (1, 0), (1, 0), (2, 0), (99, 99), (0.5, 0)], self.G, {0, 2}, 8, ZERO, INF)
        self.assertEqual(q, [self.G.vertex_at(1, 0)])

    def test_empty_guidance_is_ordinary_astar(self):
        g = RecordingGrid(3, 2)
        r, _ = run(g, (0, 0), (2, 0), [])
        self.assertEqual((r.rekeys, r.pops, g.expanded), (0, 3, [0, 1]))

    def test_deadline_even_with_candidate_queued(self):
        # Calls from guided_astar: 1 start, 2 after filter, 3 loop, 4-5 succ of s, 6 loop,
        # 7 Q advance, 8-10 succ of (1,0), 11 loop: goal is queued but the clock has hit cutoff.
        clk = TripClock("guided_astar", 1.0, nth=11)
        r, path = run(self.G, (0, 0), (2, 0), [(1, 0)], p_max=3, cutoff=1.0, clock=clk)
        self.assertEqual((r.status, path, r.pops), (Status.DEADLINE_EXCEEDED, None, 2))

    def test_clock_at_cutoff_at_entry(self):
        r, _ = run(self.G, (0, 0), (2, 0), [], cutoff=5.0, clock=lambda: 5.0)
        self.assertEqual((r.status, r.pops), (Status.DEADLINE_EXCEEDED, 0))


class FilterGuides(unittest.TestCase):
    G = GridMap(3, 3, [(1.9, 1.9, 2.1, 2.1)])  # blocks vertex (2,2)

    def f(self, raw, k_max=8, excluded=()):
        return [self.G.position(v) for v in filter_guides(raw, self.G, set(excluded), k_max, ZERO, INF)]

    def test_order_and_duplicates(self):
        self.assertEqual(self.f([(2, 1), (1, 1), (2.0, 1.0), (0, 1), (1, 1)]), [(2.0, 1.0), (1.0, 1.0), (0.0, 1.0)])

    def test_invalid_individual_entries_discarded(self):
        raw = [(math.nan, 0), (0, math.inf), (2, 2), (-1, 0), (3, 0), (0.5, 1), (10**400, 0),
               (True, 0), "ab", (1,), (1, 1, 1), None, ("1", 1), (1, 0)]
        self.assertEqual(self.f(raw, k_max=len(raw)), [(1.0, 0.0)])

    def test_start_goal_removed(self):
        self.assertEqual(self.f([(0, 0), (1, 1), (2, 1)], excluded={0, 5}), [(1.0, 1.0)])

    def test_invalid_container_disables(self):
        for raw in (None, {(1, 1): 1}, "11", 5, {(1, 1)}):
            self.assertEqual(self.f(raw), [])

    def test_k_max_on_raw_length(self):
        self.assertEqual(self.f([(1, 1)] * 3, k_max=3), [(1.0, 1.0)])
        # Oversized raw list disables all guidance even if the deduplicated list would fit.
        self.assertEqual(self.f([(1, 1)] * 4, k_max=3), [])

    def test_disabled_guidance_gives_ordinary_astar(self):
        for raw in (None, [(1, 0)] * 9):
            g = RecordingGrid(3, 2)
            r, _ = run(g, (0, 0), (2, 0), raw)
            self.assertEqual((r.status, r.rekeys, g.expanded), (Status.GEOMETRIC_CANDIDATE, 0, [0, 1]))


class SearchSemantics(unittest.TestCase):
    def check_against_oracle(self, grid, s, goal, guides, p_max=1000, expect_rekeys_matter=False):
        rg = RecordingGrid(grid.width, grid.height, grid.obstacles)
        r = guided_astar(rg, s, goal, guides, INF, p_max, clock=ZERO, k_max=64, v_max=1000)
        q = filter_guides(guides, rg, {s, goal}, 64, ZERO, INF)
        status, cost, pops, exp, reopens = oracle(rg, s, goal, q, p_max)
        self.assertEqual((r.status.value, r.cost, r.pops, rg.expanded, r.reopens), (status, cost, pops, exp, reopens))
        if expect_rekeys_matter:
            self.assertNotEqual(oracle(rg, s, goal, q, p_max, stale=True)[3], exp)
        return r, exp

    def test_rekey_reverses_order(self):
        # 4x2, s=(3,0), goal=(2,1), Q=[(3,1),(0,1)]. After expanding (3,1) the target becomes (0,1):
        # (2,0) rekeys from 1+1+sqrt2 to 1+1+sqrt5 > goal's 2+0+2, so the goal pops before (2,0).
        g = GridMap(4, 2)
        r, _ = self.check_against_oracle(g, 3, 6, [(3, 1), (0, 1)], expect_rekeys_matter=True)
        self.assertEqual((r.pops, r.rekeys, r.cost), (3, 1, 2.0))

    def test_reopen_strictly_better(self):
        # 4x4, (1,1) blocked; s=(0,1), goal=(3,0), Q=[(3,3),(2,0)]. (2,0) is first closed with g=5
        # via (2,1), later reached with g=3 via (1,0): reopened, popped again, parent updated.
        g = GridMap(4, 4, [(0.9, 0.9, 1.1, 1.1)])
        r, exp = self.check_against_oracle(g, 4, 3, [(3, 3), (2, 0)])
        self.assertEqual((r.reopens, r.pops, r.cost), (1, 13, 4.0))
        self.assertEqual([g.position(v) for v in r.path], [(0, 1), (0, 0), (1, 0), (2, 0), (3, 0)])
        self.assertEqual(exp.count(g.vertex_at(2, 0)), 2)  # repeated pop counted against P_max
        r, _ = run(g, (0, 1), (3, 0), [(3, 3), (2, 0)], p_max=12)
        self.assertEqual(r.status, Status.POP_LIMIT_REACHED)

    def test_equal_cost_does_not_reopen(self):
        # 2x3, s=(1,0), goal=(0,2), Q=[(0,1),(0,0)]: a CLOSED vertex is re-reached at equal cost.
        g = GridMap(2, 3)
        r, exp = self.check_against_oracle(g, 1, 4, [(0, 1), (0, 0)])
        self.assertEqual((r.reopens, r.pops, r.cost), (0, 5, 3.0))
        self.assertEqual(len(exp), len(set(exp)))

    def test_future_guide_expanded_out_of_order_is_skipped(self):
        # 3x4, s=(2,1), goal=(0,3), Q=[(1,2),(1,1)]: (1,1) is expanded before (1,2); when (1,2)
        # is expanded both are removed in one advance.
        g = GridMap(3, 4)
        r, exp = self.check_against_oracle(g, 5, 9, [(1, 2), (1, 1)])
        self.assertEqual((r.status, r.cost), (Status.GEOMETRIC_CANDIDATE, 4.0))
        self.assertLess(exp.index(g.vertex_at(1, 1)), exp.index(g.vertex_at(1, 2)))

    def test_unreachable_guide(self):
        # (4,2) is a free vertex but both of its edges cross thin walls: the guide is unreachable.
        walls = [thin_wall(3.4, 1.5, 3.6, 2.5), thin_wall(3.5, 1.4, 4.5, 1.6)]
        g = GridMap(5, 3, walls)
        self.assertIn(g.vertex_at(4, 2), g.free)
        r, path = run(g, (0, 0), (4, 0), [(4, 2)])
        self.assertEqual((r.status, r.cost), (Status.GEOMETRIC_CANDIDATE, 4.0))
        self.assertNotIn((4.0, 2.0), path)
        r, _ = run(g, (0, 0), (4, 2), [])
        self.assertEqual(r.status, Status.NO_PATH_IN_GRAPH)

    def test_segment_checked_not_only_endpoints(self):
        g = GridMap(3, 1, [thin_wall(0.4, -0.5, 0.6, 0.5)])
        self.assertEqual(g.free, frozenset({0, 1, 2}))
        self.assertFalse(g.valid_edge(0, 1, ZERO, INF))
        self.assertTrue(g.valid_edge(1, 2, ZERO, INF))
        self.assertFalse(GridMap(3, 1, [(1.0, 0.0, 1.5, 0.0)]).valid_edge(0, 1, ZERO, INF))  # touching = hit
        r, _ = run(g, (0, 0), (2, 0), [])
        self.assertEqual((r.status, r.pops), (Status.NO_PATH_IN_GRAPH, 1))

    def test_randomized_against_oracle(self):
        rng = random.Random(7)
        for _ in range(300):
            w, h = rng.randint(1, 6), rng.randint(1, 6)
            boxes = [(x - 0.1, y - 0.1, x + 0.1, y + 0.1) for x, y in
                     ((rng.randint(0, w - 1), rng.randint(0, h - 1)) for _ in range(rng.randint(0, 4)))]
            boxes += [thin_wall(x + 0.4, y - 0.5, x + 0.6, y + 0.5) for x, y in
                      ((rng.randint(0, w - 1), rng.randint(0, h - 1)) for _ in range(rng.randint(0, 2)))]
            g = GridMap(w, h, boxes)
            free = sorted(g.free)
            if not free:
                continue
            s, goal = rng.choice(free), rng.choice(free)
            guides = [(rng.randint(-1, w), rng.randint(-1, h)) for _ in range(rng.randint(0, 5))]
            self.check_against_oracle(g, s, goal, guides, p_max=rng.randint(1, 40))


class IndexedHeap(unittest.TestCase):
    def test_one_entry_per_vertex_and_rekey(self):
        hp = _IndexedHeap()
        for v in range(10):
            hp.insert_or_update((float(v), 0.0, v))
        hp.insert_or_update((-1.0, 0.0, 9))  # decrease
        hp.insert_or_update((20.0, 0.0, 0))  # increase also restores order
        self.assertEqual(len(hp), 10)
        hp.rekey_all(lambda v: (float(-v), 0.0, v), ZERO, INF)  # reverses order
        self.assertEqual([hp.pop_min() for _ in range(10)], list(range(9, -1, -1)))

    def test_ties_by_h_then_id(self):
        hp = _IndexedHeap()
        for k in [(1.0, 0.5, 3), (1.0, 0.5, 1), (1.0, 0.25, 7), (0.5, 9.0, 8)]:
            hp.insert_or_update(k)
        self.assertEqual([hp.pop_min() for _ in range(4)], [8, 7, 1, 3])


class Deadlines(unittest.TestCase):
    """Deadline at (==) and past (>) cutoff inside each nested operation propagates as
    DEADLINE_EXCEEDED, never as an invalid edge, disabled guidance, or NO_PATH."""

    def case(self, where, grid, s, goal, guides):
        for value in (1.0, 2.0):  # equality and overrun, cutoff = 1.0
            clk = TripClock(where, value)
            r, path = run(grid, s, goal, guides, cutoff=1.0, clock=clk)
            self.assertTrue(clk.tripped, where)
            self.assertEqual((r.status, path), (Status.DEADLINE_EXCEEDED, None), (where, value))

    def test_filtering(self):
        self.case("filter_guides", GridMap(3, 2), (0, 0), (2, 0), [(1, 0)])

    def test_edge_validation(self):
        # Every edge would otherwise be invalid -> NO_PATH; deadline must win.
        self.case("valid_edge", GridMap(2, 1, [thin_wall(0.4, -0.5, 0.6, 0.5)]), (0, 0), (1, 0), [])

    def test_edge_validation_inside_obstacle_loop(self):
        clk = TripClock("valid_edge", 1.0, nth=2)  # 1st call at entry, 2nd in the obstacle loop
        r, _ = run(GridMap(3, 2, [thin_wall(5, 5, 6, 6)]), (0, 0), (2, 0), [], cutoff=1.0, clock=clk)
        self.assertEqual(r.status, Status.DEADLINE_EXCEEDED)

    def test_rekeying(self):
        self.case("rekey_all", GridMap(3, 2), (0, 0), (2, 0), [(1, 0)])

    def test_reconstruction(self):
        self.case("reconstruct", GridMap(3, 2), (0, 0), (2, 0), [])

    def test_nested_raise_directly(self):
        with self.assertRaises(DeadlineExceeded):
            GridMap(2, 1).valid_edge(0, 1, lambda: 3.0, 3.0)


class InputsAndReconstruct(unittest.TestCase):
    G = GridMap(3, 2, [(1.9, 0.9, 2.1, 1.1)])  # (2,1) blocked

    def status(self, s, goal, **caps):
        return guided_astar(self.G, s, goal, [], INF, 10, clock=ZERO, **{**CAPS, **caps}).status

    def test_invalid_input(self):
        self.assertEqual(self.status(0, 2, v_max=4), Status.INVALID_INPUT)  # |V|=5 > 4
        self.assertEqual(self.status(0, 2, v_max=5), Status.GEOMETRIC_CANDIDATE)
        for s, goal in ((5, 2), (0, 99), (-1, 2), (True, 2), (0, 2.0), (None, 2)):
            self.assertEqual(self.status(s, goal), Status.INVALID_INPUT, (s, goal))

    def test_invalid_config_raises(self):
        for caps in ({"k_max": 0}, {"v_max": -1}, {"k_max": 1.5}):
            with self.assertRaises(ValueError):
                guided_astar(self.G, 0, 2, [], INF, 1, clock=ZERO, **{**CAPS, **caps})
        with self.assertRaises(ValueError):
            guided_astar(self.G, 0, 2, [], INF, 0, clock=ZERO, **CAPS)
        with self.assertRaises(ValueError):
            guided_astar(self.G, 0, 2, [], math.nan, 1, clock=ZERO, **CAPS)
        with self.assertRaises(ValueError):
            GridMap(2, 2, [(1, 1, 0, 0)])
        with self.assertRaises(ValueError):
            GridMap(2, 2, [(0, 0, math.inf, 1)])

    def test_nonfinite_cost_and_priority_rejected(self):
        for bad in (INF, math.nan, 1e308):  # 1e308: second edge overflows g to inf

            class Bad(GridMap):
                def cost(self, u, v, bad=bad):
                    return bad

            r = guided_astar(Bad(3, 1), 0, 2, [], INF, 10, clock=ZERO, **CAPS)
            self.assertEqual(r.status, Status.INVALID_INPUT, bad)

        class Huge(GridMap):
            def position(self, v):
                return (1e308 * (v % self.width), 0.0)

        r = guided_astar(Huge(3, 1), 0, 2, [], INF, 10, clock=ZERO, **CAPS)
        self.assertEqual(r.status, Status.INVALID_INPUT)
        self.assertIn("priority", r.detail)

    def test_reconstruct_failures(self):
        with self.assertRaises(InvalidPath):
            reconstruct({0: None, 2: 1}, 0, 2, 5, ZERO, INF)  # 1 has no parent
        with self.assertRaises(InvalidPath):
            reconstruct({0: None, 1: 2, 2: 1}, 0, 2, 5, ZERO, INF)  # cycle 1 <-> 2
        self.assertEqual(reconstruct({0: None, 1: 0, 2: 1}, 0, 2, 3, ZERO, INF), [0, 1, 2])
        with self.assertRaises(InvalidPath):
            reconstruct({0: None, 1: 0, 2: 1}, 0, 2, 2, ZERO, INF)  # 3 vertices > |V| cap


class Experiment(unittest.TestCase):
    G = GridMap(3, 2, [thin_wall(1.4, 0.5, 1.6, 1.5)])  # cuts edge (1,1)-(2,1)

    def test_validate_path(self):
        g, ids = self.G, (lambda *pts: [self.G.vertex_at(*p) for p in pts])
        self.assertEqual(validate_path(g, ids((0, 0), (1, 0), (2, 0)), 0, 2), 2.0)
        self.assertEqual(validate_path(g, [4], 4, 4), 0.0)
        for path in (ids((0, 0), (2, 0)), ids((0, 0), (1, 1)), ids((1, 0), (2, 0)), [], None, [0, 1, 99]):
            self.assertIsNone(validate_path(g, path, 0, 2), path)
        self.assertIsNone(validate_path(g, ids((1, 1), (2, 1)), 4, 5))  # crosses thin wall
        self.assertIsNone(validate_path(GridMap(3, 1, [(0.9, -1, 1.1, 1)]), [0, 1, 2], 0, 2))  # blocked vertex

    def test_split_by_map_disjoint(self):
        train, ev = split_by_map(range(10), 0.3, random.Random(0))
        self.assertEqual((len(ev), set(train) & set(ev), sorted(train + ev)), (3, set(), list(range(10))))

    def test_guidance_time_charged_against_budget(self):
        class Clock:
            t = 0.0

            def __call__(self):
                return self.t

        clk = Clock()

        def slow_provider(grid, s, goal):
            clk.t += 1.0  # guidance consumes the whole 1 s budget
            return [(1, 0)]

        base = run_method(self.G, 0, 2, None, 1.0, 10, 8, 100, clk)
        guided = run_method(self.G, 0, 2, slow_provider, 1.0, 10, 8, 100, clk)
        self.assertEqual((base["status"], base["valid_by_deadline"]), ("GEOMETRIC_CANDIDATE", True))
        self.assertEqual((guided["status"], guided["valid_by_deadline"], guided["guidance_s"]), ("DEADLINE_EXCEEDED", False, 1.0))

    def test_provider_failure_recorded_and_guidance_disabled(self):
        def broken(grid, s, goal):
            raise RuntimeError("model down")

        r = run_method(self.G, 0, 2, broken, INF, 10, 8, 100, ZERO)
        self.assertEqual((r["guidance_failure"], r["status"], r["rekeys"], r["length_m"]), ("RuntimeError", "GEOMETRIC_CANDIDATE", 0, 2.0))

    def test_validation_finishing_at_cutoff_is_not_by_deadline(self):
        clk = TripClock("run_method", 1.0, nth=4)  # 4th clock read in run_method = end of validation
        r = run_method(self.G, 0, 2, None, 1.0, 10, 8, 100, clk)
        self.assertEqual((r["status"], r["validated"], r["valid_by_deadline"]), ("GEOMETRIC_CANDIDATE", True, False))


if __name__ == "__main__":
    unittest.main()
