# Viktor: LLM-guided A* pseudocode

Date: 2026-10-03. Language-agnostic research specification only; no executable implementation. Source: Algorithm 1 and Section 3.2 of [LLM-A*, Meng et al., EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.60/), discussed in [our paper review](llm_astar_paper_review.md).

## 1. Problem definition

First experiment: generate a geometric path on a finite, known, static graph using learned intermediate waypoints to guide A* search order. Compare it with ordinary A* on identical graphs. A second pseudocode block identifies the interfaces required to place this search inside Viktor's camera-to-plan cycle.

This is an **adapted experimental variant**, not an exact reproduction of the paper. It retains the additive waypoint-distance heuristic but resolves several ambiguities:

| Detail | Selected behavior |
|---|---|
| Waypoint membership | Accept exact free graph vertices; discard off-grid, blocked, out-of-bounds, or nonfinite suggestions |
| Start/goal in guidance | Remove both from intermediate guides; keep the goal as the search objective |
| Target advancement | Advance when a target is expanded; also skip subsequent targets already CLOSED |
| Exhausted/invalid guidance | Set the added heuristic to zero; do not duplicate the goal heuristic |
| Improved path to a CLOSED state | Reopen it when its recorded cost strictly decreases |
| Termination | Return the first found goal path; no shortest-path guarantee is claimed |
| Resource exhaustion | Report a limit/deadline status, never infer that no path exists |

The first experiment does not require a driving controller or a new software framework. The learned-guidance interface is specified here; a trained Viktor model is not assumed to exist. All eventual learned components retain the user's requirement of training from random initialization.

## 2. Assumptions

1. The benchmark graph is finite, with unique stable state IDs, positive edge costs, and a frozen map snapshot. A changing graph requires a new planning cycle.
2. Benchmark positions use meters and edge costs are geometric lengths. For the examples, neighbors are four-connected grid states with one-meter edges. These are synthetic benchmark choices, not Nissan Leaf parameters.
3. Every accepted transition has a collision-free connecting segment, not merely free endpoints. If a finite footprint is represented by obstacle inflation, that footprint/inflation model must be declared. The basic point-graph benchmark does not establish vehicle feasibility.
4. Model outputs and graph size have finite declared caps. A malformed or oversized guide list disables guidance rather than initiating unbounded parsing or retries.
5. Fresh structured evidence, decisions, and reasons remain mandatory inside Viktor's 20-30 ms cycle. Invalid required reasoning cannot be repaired by silently running A* alone and reporting success.
6. Deadline checks detect expiration. Hard timing guarantees additionally require bounds on camera behavior, neural calls, graph construction, queue rebuilding, trajectory generation, validation, and publication. No such measurements are supplied by this pseudocode.

## 3. Notation

| Symbol/name | Role and meaning | Shape/domain | Units |
|---|---|---|---|
| G, V, E | Input graph, free vertices, candidate edges | Finite sets; fixed per search | Counts dimensionless |
| s, goal, u, v | Start, goal, and search vertices | Vertex IDs with associated position p in R² | IDs dimensionless; position m |
| raw_guides, Q | Model proposals and filtered ordered intermediate guides | At most K_max vertices/coordinates | IDs dimensionless; coordinates m |
| g[v], parent[v] | Best discovered cost and predecessor | Nonnegative cost or infinity; vertex ID or NONE | m; dimensionless ID |
| h(v), b(v), f(v) | Goal heuristic, guide bias, and queue priority | Finite scalars for admitted vertices | m |
| OPEN, CLOSED | Indexed priority queue and expanded-state set | At most one OPEN entry per vertex | Dimensionless counts |
| pops, P_max | Queue-pop count and its configured cap | Nonnegative count; positive integer cap | Dimensionless |
| K_max, V_max | Admitted guide and graph-size caps | Positive integers, to be selected experimentally | Dimensionless |
| t0, d, D | Earliest contributing camera exposure start, absolute deadline, cycle allowance | Monotonic timestamps; D = 0.020 or 0.030 in latency trials | s |
| d_search, B_finish | Search cutoff and time reserved for trajectory, validation, publication | Timestamp; positive duration | s |
| scene, R | Estimated scene and structured evidence/decision/reasons | Bounded typed records | Each physical attribute declares SI units; flags/IDs dimensionless |
| theta | Viktor's learned parameters | Architecture-dependent tensors | Computational parameters dimensionless |
| path, trajectory | Geometric vertex sequence and time-parameterized candidate | Bounded sequences | Positions m; trajectory times s and derived quantities in SI units |

## 4. Derivation

Ordinary A* orders states by discovered start cost plus goal-distance estimate. The added distance to the next intermediate guide changes search order. Advancing that guide changes the priority function for every OPEN state, requiring a complete queue rekey.

The added distance can overestimate the actual remaining cost. Reopening improved CLOSED states repairs discovered-cost propagation; it does not make the heuristic admissible or restore an optimality guarantee. When guidance is empty from the start, the core reduces to ordinary A* under the graph and heuristic assumptions above.

## 5. Final equations

For Euclidean-distance costs:

$$
h(v)=\|p(v)-p(goal)\|_2,\qquad
b(v)=\begin{cases}\|p(v)-p(Q_0)\|_2,&Q\ne\varnothing,\\0,&Q=\varnothing,\end{cases}
$$

$$
f(v)=g[v]+h(v)+b(v),\qquad
\operatorname{key}(v)=(f(v),h(v),\operatorname{ID}(v)).
$$

Keys compare lexicographically; the last two fields specify deterministic ties. Relaxation uses a strict improvement: replace g[v] only when g[u] + cost(u,v) < g[v]. Reject nonfinite costs; do not hide them with arbitrary clamping.

$$
d=t_0+D,\qquad d_{search}=d-B_{finish}.
$$

All camera, model, graph, search, trajectory, validation, and publication work is charged to the same deadline. B_finish is a budget to validate experimentally, not a proven runtime bound.

## 6. Constraints and operation contracts

- `CHECK_TIME(cutoff)` returns `DEADLINE_EXCEEDED` from the current operation if the monotonic clock is at or past the cutoff. Nested operations propagate that status. Check before and after expensive calls and during long loops. An uninterruptible overrun remains a deadline miss.
- `FILTER_GUIDES` preserves order, accepts only exact free graph vertices, removes start/goal/all duplicate entries, and enforces K_max. Invalid individual entries are discarded; an invalid container or oversized list yields an empty guide list. No snapping radius is invented.
- `VALID_EDGE` checks bounds and the whole transition against the frozen map. It is applied even when both endpoints are free. A failure to finish checking before the cutoff is not a valid edge.
- `REKEY_ALL` updates every existing OPEN entry for the new guide, then restores the queue order. It never leaves old-guide priorities mixed with new ones.
- `RECONSTRUCT` follows parent links back to s with a length cap of |V|; missing parents or a cycle produce `INVALID_PATH`. Its output is a geometric candidate, not permission to actuate.
- `PUBLISH_BEFORE` succeeds only if the validated payload becomes available to its consumer before the publication deadline. It reports a deadline miss otherwise. Payloads separately carry acquisition timestamps and a trajectory-validity interval established by validation; consumers enforce that interval and the freshness contract. The inference deadline is not the trajectory expiration time. This is an interface requirement awaiting implementation and measurement.

## 7. Algorithm / pseudocode

### A. Bounded guided search

```text
FUNCTION GUIDED_ASTAR(G, s, goal, raw_guides, cutoff, P_max):
    CHECK_TIME(cutoff)
    IF G exceeds V_max OR s/goal are invalid:
        RETURN INVALID_INPUT

    Q <- FILTER_GUIDES(raw_guides, G, excluding={s, goal}, cutoff)
    CHECK_TIME(cutoff)
    g[*] <- infinity; g[s] <- 0
    parent[s] <- NONE
    CLOSED <- empty set
    OPEN <- indexed min-priority queue containing (s, KEY(s, Q, g))
    pops <- 0

    WHILE OPEN is not empty:
        CHECK_TIME(cutoff)
        IF pops >= P_max:
            RETURN POP_LIMIT_REACHED

        u <- OPEN.REMOVE_MIN()
        pops <- pops + 1

        IF u == goal:
            path <- RECONSTRUCT(parent, s, goal, cutoff)
            CHECK_TIME(cutoff)
            RETURN GEOMETRIC_CANDIDATE(path, pops)

        ADD u TO CLOSED
        old_target <- FIRST(Q) or NONE
        WHILE Q is not empty AND FIRST(Q) is in CLOSED:
            CHECK_TIME(cutoff)
            REMOVE_FIRST(Q)
        IF (FIRST(Q) or NONE) != old_target:
            REKEY_ALL(OPEN, using KEY(vertex, Q, g), cutoff)

        FOR each successor v of u in stable ID order:
            CHECK_TIME(cutoff)
            IF NOT VALID_EDGE(G, u, v, cutoff):
                CONTINUE
            candidate_cost <- g[u] + cost(u, v)
            IF candidate_cost < g[v]:
                g[v] <- candidate_cost
                parent[v] <- u
                REMOVE v FROM CLOSED if present
                OPEN.INSERT_OR_DECREASE(v, KEY(v, Q, g))

    CHECK_TIME(cutoff)
    RETURN NO_PATH_IN_GRAPH
```

All returns from nested operations propagate failure. `KEY` is exactly the equation in Section 5. The queue contains one live entry per vertex, so a rekey cannot expose stale duplicate entries. Count repeated pops after reopening against P_max, including the goal pop. The budget check precedes the pop; a goal found on pop P_max is permitted.

### B. Proposed Viktor cycle: interfaces, not implemented modules

```text
FUNCTION VIKTOR_CYCLE(capture, ego_state, instruction, theta, D):
    t0 <- earliest exposure-start timestamp in this admitted camera bundle
    deadline <- t0 + D
    search_cutoff <- deadline - B_finish

    images, calibration, timestamps <- ACQUIRE_AND_PREPROCESS(capture, deadline)
    CHECK_TIME(deadline)

    scene, R, raw_guides <- VIKTOR_FORWARD(
        images, calibration, timestamps, ego_state, instruction, theta)
    CHECK_TIME(deadline)
    IF required scene evidence or R is invalid/stale:
        RETURN NO_VALIDATED_PLAN(INVALID_EVIDENCE_OR_REASONING)

    G, start, local_goal <- BUILD_BOUNDED_LOCAL_GRAPH(scene, ego_state, deadline)
    CHECK_TIME(search_cutoff)
    result <- GUIDED_ASTAR(G, start, local_goal, raw_guides, search_cutoff, P_max)
    IF result is not GEOMETRIC_CANDIDATE:
        RETURN NO_VALIDATED_PLAN(result.status)

    trajectory <- MAKE_TIME_PARAMETERIZED_TRAJECTORY(
        result.path, ego_state, scene, deadline)
    CHECK_TIME(deadline)
    validation <- VALIDATE_TRAJECTORY_AND_REASONS(trajectory, R, scene, deadline)
    IF NOT validation.accepted:
        RETURN NO_VALIDATED_PLAN(VALIDATION_FAILED)

    RETURN PUBLISH_BEFORE(
        trajectory, R, source_timestamps=timestamps,
        publish_by=deadline, validity_interval=validation.validity_interval)
```

`VIKTOR_FORWARD` is the proposed bounded shared perception/reasoning/guidance computation, not a call to a pretrained external model. R includes evidence references and validity, a maneuver decision, and structured reasons. Guidance should use the declared lattice vertices so off-grid outputs can be rejected deterministically. Invalid optional guides can disable the search bias; invalid mandatory R cannot.

`BUILD_BOUNDED_LOCAL_GRAPH`, `MAKE_TIME_PARAMETERIZED_TRAJECTORY`, and `VALIDATE_TRAJECTORY_AND_REASONS` are **unresolved contracts**, not hidden solved problems. They require a coordinate frame, occupancy/uncertainty policy, local goal/horizon, vehicle footprint, dynamic predictions, and physical limits. The validator must check dynamic collision avoidance, lane containment, relevant following constraints, speed/acceleration/curvature limits, and consistency between the actual plan and reported decision/reasons. A waypoint that search bypassed cannot be reported as an executed maneuver. These checks do not prove causal faithfulness of the model's reasoning.

Failures are passed to the separately specified supervisor; this document does not invent a braking command or reuse an old plan without revalidation. The geometric benchmark can be implemented and evaluated independently of these driving interfaces. The wrapper does not establish that their combined cost fits 20-30 ms.

### C. Fair first experiment

```text
FOR each held-out static map and start/goal pair:
    Obtain bounded learned guide proposals; record guidance latency/failures
    Run ordinary A* core with raw_guides = empty
    Run guided core with the learned proposals
    Use the same graph, costs, collision checks, ties, and search caps
    Independently validate each returned path and recompute its actual length
    Record guidance + filtering + search + rekey + final-check wall time
    Record peak total memory, queue pops, path length, and every failure status
```

Ordinary A* incurs no guidance-model cost; count that extra cost in the guided total. For a deadline-limited comparison, each method receives the same total deadline from the same defined origin, so time spent on guidance reduces the remaining search time. Report uncapped/reference shortest-path results separately if a baseline misses its operational deadline. Split by map, not just start/goal pairs, to avoid sharing the same map between training and evaluation. Report valid-plan-by-deadline coverage over all admitted tasks; do not omit failed or slow cases.

## 8. Example calculations

Use an obstacle-free 3-by-2 vertex grid: x in {0,1,2}, y in {0,1}, four-connected edges of length 1 m, ID(x,y)=3y+x, start (0,0), goal (2,0).

With guide (1,0), the popped keys are (3,2,0) at the start, then (2,1,1) at (1,0). Expanding the guide exhausts Q and clears the bias. The goal then has key (2,0,2), and the returned path is [(0,0),(1,0),(2,0)] with length 2 m and three pops.

With guide (1,1), after expanding (1,0), the goal and guide both have f=2+sqrt(2). Their h values are 0 and sqrt(2), respectively, so the goal wins the tie. The same path is returned without visiting the guide: guidance remains a soft preference.

## 9. Expected outputs

The researcher independently checked the trace. Scalar distances and priorities were checked in an ephemeral arithmetic-only shell calculation; no implementation was run.

| Input variation | Expected result |
|---|---|
| Guide (1,0), P_max=3, sufficient time | Candidate [(0,0),(1,0),(2,0)], length 2 m, pops=3 |
| Guide (1,1), P_max=3, sufficient time | Same candidate and length; guide need not be visited |
| Either case with P_max=2 | POP_LIMIT_REACHED before the goal pop; no validated path result |
| Raw guides [start,(1,0),(1,0),goal,(99,99),(0.5,0)] | Q=[(1,0)]; out-of-bounds and off-grid suggestions discarded |
| Empty valid guidance | Ordinary A* priority from initialization |
| Start equals goal, P_max>=1, sufficient time | One-vertex path, length 0, one pop |
| Goal disconnected, frontier exhausted before limits | NO_PATH_IN_GRAPH |
| Clock at or beyond cutoff | DEADLINE_EXCEEDED, even if a candidate remains queued |

The numeric equality f=2+sqrt(2)=3.414213562373095 can be reproduced to absolute tolerance 1e-12. Implementations need a declared deterministic floating-point ordering policy; no epsilon-based approximate comparator is specified here.

## 10. Potential mathematical risks

This variant has no general shortest-path or hard-real-time guarantee. Reopening can increase work; a fixed pop cap limits exploration, not the cost of each expansion. Guidance may increase search effort or be discarded. The graph may omit a feasible continuous path, and its candidate may fail subsequent vehicle validation. Poor perception or prediction invalidates a correct search over the wrong graph.

An operation bound for bounded successor degree b is O(V_max + K_max + P_max * b * log V_max + J * V_max), excluding collision-check cost, neural inference, graph construction, and final processing. This includes record initialization and guide filtering with constant-time vertex lookup/deduplication; J <= K_max counts target changes and assumes linear-time full heap rebuilds. Search storage is O(V_max) with one OPEN entry and one record per vertex. Total system memory also includes graph representation and all neural resources.

## 11. Results in limiting / edge cases

- Guidance empty from the beginning: ordinary A* behavior under the stated assumptions.
- Guides exhausted midway: priority becomes g+h, but do not infer optimality of an already terminated guided run.
- Unreachable guide: search may retain an unhelpful bias; it can still find the goal without that guide if budget permits.
- A future guide expanded out of order: skip it when it reaches the front of Q and is already CLOSED.
- Better route to a CLOSED vertex: reopen it and count its next pop again.
- Invalid optional guidance: continue with zero bias. Invalid mandatory scene reasoning: fail the Viktor cycle explicitly.
- Budget/deadline exhausted: unknown whether a valid path exists; this is distinct from exhausting the graph.

## Open questions for the user

No clarification is needed for this pseudocode. Before vehicle implementation, choose and measure the camera configuration, graph resolution/extent, guide/pop caps, local goal definition, dynamics/prediction contracts, B_finish, and supervisor behavior. The timing-origin assumption remains earliest contributing exposure start to published validated output; arbitrary-event observation delay and actuator response need separate accounting as described in [the latency note](low_latency_architecture.md).
