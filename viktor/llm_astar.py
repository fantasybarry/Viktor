"""Bounded geometric LLM-guided A* (adapted LLM-A* variant).

Specification: docs/math/llm_astar_pseudocode.md (2026-10-03),
Sections 5 (equations), 6 (operation contracts), 7A (GUIDED_ASTAR).

Pure mathematical core: no file, network, or logging I/O. The monotonic
clock is injected (``clock``) so deadline behavior is deterministic in tests.

Benchmark graph (spec Section 2, declared point-graph assumptions):
  * Vertices are lattice points (x, y) in meters, x in {0..W-1}, y in {0..H-1},
    stable ID(x, y) = W*y + x. Frame: map frame, units m.
  * The frozen map is a tuple of closed axis-aligned rectangles (m). A vertex
    is free iff it lies in no rectangle; V is the set of free vertices.
  * Candidate edges E are 4-connected neighbors in V; cost = Euclidean length
    (1 m). VALID_EDGE checks map bounds and the whole closed segment against
    every rectangle, even when both endpoints are free.
  * Point robot: no footprint and no obstacle inflation. This establishes no
    vehicle feasibility.

Deterministic float ordering policy (spec Section 9): IEEE-754 binary64,
f evaluated left to right as (g + h) + b, h and b via math.dist, keys
compared exactly as Python tuples (f, h, ID). No epsilon comparator and no
clamping. NaN/inf keys or costs are rejected (INVALID_INPUT), never ordered.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

Clock = Callable[[], float]
"""Monotonic clock in seconds, same timeline as every cutoff."""

Rect = tuple[float, float, float, float]
"""Closed obstacle box (x_min, y_min, x_max, y_max) in meters, map frame."""

Key = tuple[float, float, int]
"""(f, h, ID): priority f (m), goal heuristic h (m), stable vertex ID."""


class Status(Enum):
    INVALID_INPUT = "INVALID_INPUT"  # graph over V_max, invalid start/goal, nonfinite cost/key
    POP_LIMIT_REACHED = "POP_LIMIT_REACHED"  # P_max pops used; path existence unknown
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"  # clock >= cutoff; path existence unknown
    NO_PATH_IN_GRAPH = "NO_PATH_IN_GRAPH"  # frontier exhausted within all limits
    INVALID_PATH = "INVALID_PATH"  # reconstruction hit a missing parent or a cycle
    GEOMETRIC_CANDIDATE = "GEOMETRIC_CANDIDATE"  # candidate only, not permission to actuate


@dataclass(frozen=True)
class SearchResult:
    """Outcome of guided_astar.

    path: vertex IDs start..goal (only for GEOMETRIC_CANDIDATE), else None.
    cost: g[goal] in m (only for GEOMETRIC_CANDIDATE), else None.
    pops: queue pops counted against P_max, including repeated and goal pops.
    rekeys: number of REKEY_ALL calls (J in spec Section 10).
    reopens: CLOSED vertices reopened by a strictly better cost.
    """

    status: Status
    path: tuple[int, ...] | None = None
    cost: float | None = None
    pops: int = 0
    rekeys: int = 0
    reopens: int = 0
    detail: str = ""


class DeadlineExceeded(Exception):
    """Raised by check_time; propagates through every nested operation."""


class InvalidPath(Exception):
    """Raised by reconstruct on a missing parent or a cycle."""


class _NonFinite(Exception):
    """A cost or priority was NaN or infinite."""


def check_time(clock: Clock, cutoff: float) -> None:
    """CHECK_TIME (spec Section 6): expired if the clock is at or past cutoff."""
    if clock() >= cutoff:
        raise DeadlineExceeded


class GridMap:
    """Frozen 4-connected lattice graph with closed rectangular obstacles.

    width, height: vertex counts along x and y (positive ints); spacing 1 m.
    obstacles: closed boxes (x_min, y_min, x_max, y_max), finite, in m.
    """

    def __init__(self, width: int, height: int, obstacles: Sequence[Rect] = ()) -> None:
        for name, n in (("width", width), ("height", height)):
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ValueError(f"{name} must be a positive int, got {n!r}")
        boxes = tuple(tuple(float(c) for c in r) for r in obstacles)
        for r in boxes:
            if len(r) != 4 or not all(math.isfinite(c) for c in r) or r[0] > r[2] or r[1] > r[3]:
                raise ValueError(f"obstacle must be finite (x_min, y_min, x_max, y_max) with min <= max, got {r!r}")
        self.width = width
        self.height = height
        self.obstacles: tuple[Rect, ...] = boxes  # type: ignore[assignment]
        self.free: frozenset[int] = frozenset(
            v for v in range(width * height) if not any(_point_in_box(self.position(v), r) for r in boxes)
        )

    def position(self, v: int) -> tuple[float, float]:
        """p(v) in meters, map frame."""
        return (float(v % self.width), float(v // self.width))

    def vertex_at(self, x: float, y: float) -> int | None:
        """ID of the free vertex at exactly (x, y) m, else None. No snapping."""
        try:
            fx, fy = float(x), float(y)
        except OverflowError:  # int too large for a float: out of bounds
            return None
        if not (math.isfinite(fx) and math.isfinite(fy) and fx.is_integer() and fy.is_integer()):
            return None
        xi, yi = int(fx), int(fy)
        if not (0 <= xi < self.width and 0 <= yi < self.height):
            return None
        v = yi * self.width + xi
        return v if v in self.free else None

    def successors(self, u: int) -> list[int]:
        """Candidate 4-neighbors of u in V, in ascending (stable) ID order."""
        x, y = u % self.width, u // self.width
        out = []
        if y > 0:
            out.append(u - self.width)
        if x > 0:
            out.append(u - 1)
        if x < self.width - 1:
            out.append(u + 1)
        if y < self.height - 1:
            out.append(u + self.width)
        return [v for v in out if v in self.free]

    def cost(self, u: int, v: int) -> float:
        """cost(u, v): geometric edge length in m (spec Section 2)."""
        return math.dist(self.position(u), self.position(v))

    def valid_edge(self, u: int, v: int, clock: Clock, cutoff: float) -> bool:
        """VALID_EDGE (spec Section 6): bounds plus whole-segment collision check.

        A deadline raises DeadlineExceeded; it is never reported as an invalid edge.
        """
        check_time(clock, cutoff)
        a, b = self.position(u), self.position(v)
        for px, py in (a, b):
            if not (0.0 <= px <= self.width - 1 and 0.0 <= py <= self.height - 1):
                return False
        for box in self.obstacles:
            check_time(clock, cutoff)
            if _segment_hits_box(a, b, box):
                return False
        check_time(clock, cutoff)
        return True


def _point_in_box(p: tuple[float, float], r: Rect) -> bool:
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def _segment_hits_box(a: tuple[float, float], b: tuple[float, float], r: Rect) -> bool:
    """Closed segment a-b intersects closed box r (Liang-Barsky slab clipping).

    Parametrize P(t) = a + t (b - a), t in [0, 1]; intersect the t-intervals in
    which each coordinate lies inside the box's slab. Touching counts as a hit.
    """
    t_lo, t_hi = 0.0, 1.0
    for axis in (0, 1):
        lo, hi = r[axis], r[axis + 2]
        p, d = a[axis], b[axis] - a[axis]
        if d == 0.0:
            if p < lo or p > hi:
                return False
            continue
        t1, t2 = (lo - p) / d, (hi - p) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_lo, t_hi = max(t_lo, t1), min(t_hi, t2)
        if t_lo > t_hi:
            return False
    return True


def _guide_vertex(grid: GridMap, entry: object) -> int | None:
    """One raw guide (x, y) in m -> exact free vertex ID, or None if invalid."""
    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
        return None
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in entry):
        return None
    return grid.vertex_at(entry[0], entry[1])


def filter_guides(
    raw_guides: object, grid: GridMap, excluded: set[int], k_max: int, clock: Clock, cutoff: float
) -> list[int]:
    """FILTER_GUIDES (spec Section 6) -> ordered guide list Q of vertex IDs.

    raw_guides must be a list/tuple of (x, y) coordinates in m. Any other
    container, or more than k_max raw entries, disables guidance (returns []).
    Invalid individual entries (malformed, nonfinite, off-grid, out of bounds,
    blocked) are discarded; start/goal (``excluded``) and repeat occurrences
    are removed, keeping the first occurrence; order is preserved.
    """
    if not isinstance(raw_guides, (list, tuple)) or len(raw_guides) > k_max:
        return []
    seen = set(excluded)
    q: list[int] = []
    for entry in raw_guides:
        check_time(clock, cutoff)
        v = _guide_vertex(grid, entry)
        if v is not None and v not in seen:
            seen.add(v)
            q.append(v)
    return q


class _IndexedHeap:
    """Binary min-heap of unique keys (f, h, ID) with an ID -> index map.

    Exactly one live entry per vertex, so a rekey can never expose stale keys.
    """

    def __init__(self) -> None:
        self._a: list[Key] = []
        self._pos: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._a)

    def insert_or_update(self, key: Key) -> None:
        v = key[2]
        i = self._pos.get(v)
        if i is None:
            self._a.append(key)
            i = len(self._a) - 1
            self._pos[v] = i
        else:
            self._a[i] = key
        self._sift_down(self._sift_up(i))

    def pop_min(self) -> int:
        a = self._a
        top = a[0]
        last = a.pop()
        del self._pos[top[2]]
        if a:
            a[0] = last
            self._pos[last[2]] = 0
            self._sift_down(0)
        return top[2]

    def rekey_all(self, key_of: Callable[[int], Key], clock: Clock, cutoff: float) -> None:
        """REKEY_ALL: recompute every key, then rebuild heap order in O(n)."""
        a = self._a
        for i in range(len(a)):
            check_time(clock, cutoff)
            a[i] = key_of(a[i][2])
        heapq.heapify(a)  # keys are unique tuples, so heapq order is the key order
        self._pos = {k[2]: i for i, k in enumerate(a)}
        check_time(clock, cutoff)

    def _swap(self, i: int, j: int) -> None:
        a = self._a
        a[i], a[j] = a[j], a[i]
        self._pos[a[i][2]] = i
        self._pos[a[j][2]] = j

    def _sift_up(self, i: int) -> int:
        while i > 0 and self._a[i] < self._a[(i - 1) // 2]:
            self._swap(i, (i - 1) // 2)
            i = (i - 1) // 2
        return i

    def _sift_down(self, i: int) -> None:
        a, n = self._a, len(self._a)
        while True:
            c = 2 * i + 1
            if c >= n:
                return
            if c + 1 < n and a[c + 1] < a[c]:
                c += 1
            if not a[c] < a[i]:
                return
            self._swap(i, c)
            i = c


def reconstruct(
    parent: dict[int, int | None], s: int, goal: int, vertex_count: int, clock: Clock, cutoff: float
) -> list[int]:
    """RECONSTRUCT (spec Section 6): follow parents goal -> s, at most |V| vertices.

    Raises InvalidPath on a missing parent or a cycle (detected by the |V| cap).
    """
    path = [goal]
    v = goal
    while v != s:
        check_time(clock, cutoff)
        if len(path) >= vertex_count:
            raise InvalidPath("parent chain exceeds |V| vertices (cycle)")
        nxt = parent.get(v)
        if nxt is None:
            raise InvalidPath(f"missing parent for vertex {v}")
        v = nxt
        path.append(v)
    path.reverse()
    return path


def guided_astar(
    grid: GridMap,
    s: int,
    goal: int,
    raw_guides: object,
    cutoff: float,
    p_max: int,
    *,
    k_max: int,
    v_max: int,
    clock: Clock,
) -> SearchResult:
    """GUIDED_ASTAR (spec Section 7A). Empty raw_guides gives ordinary A*.

    s, goal: vertex IDs. raw_guides: list/tuple of (x, y) m (see filter_guides).
    cutoff: absolute time on ``clock``'s timeline (s); math.inf for uncapped.
    p_max, k_max, v_max: positive int caps (spec Section 3).

    Returns the first goal path found. No shortest-path guarantee when guides
    are non-empty (b can overestimate remaining cost; spec Section 4).
    """
    for name, n in (("p_max", p_max), ("k_max", k_max), ("v_max", v_max)):
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"{name} must be a positive int, got {n!r}")
    if not isinstance(cutoff, (int, float)) or math.isnan(cutoff):
        raise ValueError(f"cutoff must be a non-NaN number, got {cutoff!r}")

    pops = rekeys = reopens = 0
    try:
        check_time(clock, cutoff)
        if len(grid.free) > v_max:
            return SearchResult(Status.INVALID_INPUT, detail=f"|V|={len(grid.free)} exceeds V_max={v_max}")
        for name, v in (("start", s), ("goal", goal)):
            if not isinstance(v, int) or isinstance(v, bool) or v not in grid.free:
                return SearchResult(Status.INVALID_INPUT, detail=f"{name} {v!r} is not a free vertex")

        q = filter_guides(raw_guides, grid, {s, goal}, k_max, clock, cutoff)
        check_time(clock, cutoff)
        qi = 0  # Q is q[qi:]; REMOVE_FIRST advances qi
        goal_pos = grid.position(goal)
        g: dict[int, float] = {s: 0.0}  # missing entry means g = +infinity
        parent: dict[int, int | None] = {s: None}
        closed: set[int] = set()

        def key(v: int) -> Key:
            # KEY (spec Section 5): f = g + h + b, evaluated as (g + h) + b.
            p = grid.position(v)
            h = math.dist(p, goal_pos)
            b = math.dist(p, grid.position(q[qi])) if qi < len(q) else 0.0
            f = g[v] + h + b
            if not math.isfinite(f):
                raise _NonFinite(f"nonfinite priority f={f!r} at vertex {v}")
            return (f, h, v)

        open_ = _IndexedHeap()
        open_.insert_or_update(key(s))

        while len(open_):
            check_time(clock, cutoff)
            if pops >= p_max:
                return SearchResult(Status.POP_LIMIT_REACHED, pops=pops, rekeys=rekeys, reopens=reopens)
            u = open_.pop_min()
            pops += 1

            if u == goal:
                path = reconstruct(parent, s, goal, len(grid.free), clock, cutoff)
                check_time(clock, cutoff)
                return SearchResult(
                    Status.GEOMETRIC_CANDIDATE, tuple(path), g[goal], pops, rekeys, reopens
                )

            closed.add(u)
            old_target = q[qi] if qi < len(q) else None
            while qi < len(q) and q[qi] in closed:
                check_time(clock, cutoff)
                qi += 1
            if (q[qi] if qi < len(q) else None) != old_target:
                open_.rekey_all(key, clock, cutoff)
                rekeys += 1

            for v in grid.successors(u):
                check_time(clock, cutoff)
                if not grid.valid_edge(u, v, clock, cutoff):
                    continue
                c = grid.cost(u, v)
                candidate_cost = g[u] + c
                if not (math.isfinite(c) and math.isfinite(candidate_cost)):
                    raise _NonFinite(f"nonfinite cost on edge {u}->{v}")
                if candidate_cost < g.get(v, math.inf):
                    g[v] = candidate_cost
                    parent[v] = u
                    if v in closed:
                        closed.remove(v)
                        reopens += 1
                    open_.insert_or_update(key(v))

        check_time(clock, cutoff)
        return SearchResult(Status.NO_PATH_IN_GRAPH, pops=pops, rekeys=rekeys, reopens=reopens)
    except DeadlineExceeded:
        return SearchResult(Status.DEADLINE_EXCEEDED, pops=pops, rekeys=rekeys, reopens=reopens)
    except InvalidPath as e:
        return SearchResult(Status.INVALID_PATH, pops=pops, rekeys=rekeys, reopens=reopens, detail=str(e))
    except _NonFinite as e:
        return SearchResult(Status.INVALID_INPUT, pops=pops, rekeys=rekeys, reopens=reopens, detail=str(e))
