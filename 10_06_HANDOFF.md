# Viktor research handoff — 2026-10-06

## 1. Resume here

The conversation is ending after mathematical research and documentation. **The current workspace contains research Markdown, not an implemented or trained Viktor model.** The latest completed technical work compared LLM-A* with the local Alpamayo 2 Super repository and formatted the comparison's formulas. The current user request is to save this handoff as **10_06_HANDOFF.md**.

Read this file first, then the six documents linked below. Preserve the distinction between user requirements, proposed designs, paper results, source-code findings, and measured results. **No vehicle/model latency or driving performance was measured.**

Do not start implementation merely because this handoff lists future development work. The user previously assigned this agent mathematical documentation and implementation to other agents. A later request to call the developer did not produce a completed implementation; do not report that it did. The latest task is this handoff.

## 2. Workspace and repository state

Verified on 2026-10-06:

| Item | State |
|---|---|
| Current workspace | /home/barry/orca/workspaces/Viktor/Research |
| Git branch | fantasybarry/Research |
| Workspace HEAD | 9173928 |
| Status before this handoff | Untracked docs/ directory |
| Existing research artifacts | Six Markdown files in docs/math/ |
| Current executable implementation/tests | None in this workspace |
| Original project overview | /home/barry/Viktor/README.md |
| Inspected Alpamayo repository | /home/barry/alpamayo2 |
| Alpamayo HEAD | beb2977d9a7e9d66837d4a3ad5144ff59de37519 |
| Alpamayo working-tree modification | Pre-existing SECURITY.md change; left untouched |

The research files have not been committed. This handoff is also a new file. Do not reset, clean, or overwrite these untracked artifacts.

Historical implementation attempts and passing-test reports are not current deliverables: the former trajectory module, tests, and review artifact are absent. Inspect the current tree before relying on old implementation claims.

Researcher and reviewer agents assisted with source inspection and mathematical checks. A developer agent was available, but no implementation is running as part of this handoff. Agent sessions may not survive into the next conversation; the files are the durable record.

## 3. Confirmed user requirements

| Topic | Confirmed requirement |
|---|---|
| Project | Viktor: own multimodal LLM/VLA for planning, prediction, following, reasoning, and embodied behavior |
| Meaning of “our own LLM” | Build and train learned components from random initialization; do not substitute pretrained weights or adapters |
| First milestone | Highway lane keeping and vehicle following |
| Vehicle | 2025 Nissan Leaf SV Plus EVs |
| Vision | Monocular/fisheye cameras; exact count and arrangement unset |
| Training hardware | Jetson AGX Orin 64GB; Thor and rented cloud GPUs are options |
| Deployment hardware | Jetson AGX Thor, 128GB memory |
| Available data | User reports synchronized camera recordings and vehicle-state logs |
| End-to-end target | 20–30 ms, including camera input and reasoning |
| Required reasoning | Structured scene evidence, maneuver decision, and reasons |
| Deliverables from this research work | Mathematical findings and pseudocode in Markdown with properly formatted formulas |

User-provided Thor specifications include 2560 CUDA cores, 96 Tensor Cores, up to 2070 sparse FP4 TFLOPS / 1035 sparse FP8 TFLOPS, a 14-core Arm CPU, and 273 GB/s memory bandwidth. These are hardware specifications, not achieved throughput for our workload.

The user wants fresh reasoning inside the same deadline. Moving required reasoning into an asynchronous explanation process does not satisfy the requirement.

The latency note provisionally treats 30 ms as the primary allowance and 20 ms as a stretch allocation, measured from exposure start/trigger to reasoning plus validated-plan output. **The precise timing origin and statistical versus hard-deadline acceptance criterion remain unresolved.** Physical actuator response is a separate interval to define and measure.

Do not add lane changes or a complete general driving stack to the initial milestone without a corresponding scope decision.

## 4. Completed tasks and durable artifacts

| Document | What is complete | Limits |
|---|---|---|
| [Highway training pilot](docs/math/highway_training_pilot.md) | Ego-frame target transform, normalized masked position loss, derivatives, numerical fixtures, precision/invalid-data requirements | Diagnostic math only; no audited dataset or executable training core |
| [Low-latency architecture](docs/math/low_latency_architecture.md) | Proposed synchronous structured reasoning architecture, timing definitions, critical-path worksheet, core FLOP/parameter and bandwidth calculations | Architecture and stage budgets are hypotheses, not a selected or benchmarked model |
| [LLM-A* paper review](docs/math/llm_astar_paper_review.md) | Paper analysis, benchmark transcription, heuristic critique, constraints, and a conditional suboptimality derivation | Static-map results do not establish driving behavior; remaining LaTeX issues are listed below |
| [LLM-A* pseudocode](docs/math/llm_astar_pseudocode.md) | Bounded graph-search experiment and proposed camera-to-plan interfaces | Explicitly an adapted algorithm; no executable search or driving interfaces |
| [Efficiency estimate](docs/math/llm_astar_efficiency_estimate.md) | Guidance-overhead break-even equations and whole-pipeline sensitivity calculations | All millisecond examples are assumed; no hardware forecast or benchmark |
| [LLM-A* versus Alpamayo 2](docs/math/llm_astar_vs_alpamayo2.md) | Local source analysis, flow matching, kinematic rollout, training/CFG distinctions, computational comparison, limitations | Applies to the recorded checkout; no weights loaded or inference run |

The latest comparison export contains 23 display-math blocks, 114 inline math expressions, and equation tags 1–21. Its units and long equations were reformatted, and delimiter, brace, environment, numbering, and table checks passed. These were source-format checks, not a visual rendering test.

Two supplied papers remain available:

- /home/barry/Downloads/2024.findings-emnlp.60.pdf — the LLM-A* paper, covered by the review and comparison.
- /home/barry/Downloads/2511.00088v2.pdf — an earlier requested paper. No standalone review for it appears among the six current files. Do not claim a complete durable review from the current artifact inventory alone; revisit the paper/history if it becomes relevant.

The original project README was read. It describes broad VLA, perception, planning, and robotics ambitions but supplies no detailed architecture, data schema, or implementation.

## 5. Mathematical findings to preserve

### Highway training diagnostic

For world-frame future position $p_j^W$, anchor position $p_0^W$, and anchor yaw $\psi_0$, the proposed ego-frame target is:

$$
y_j=R(\psi_0)^\top(p_j^W-p_0^W).
$$

Here $R$ is the planar rotation matrix; positions are in meters. With valid index set $\mathcal V$, $n=\lvert\mathcal V\rvert>0$, prediction $\widehat y$, and positive coordinate scales $s_d$:

$$
\mathcal L
=\frac{1}{2n}
\sum_{(b,j)\in\mathcal V}
\sum_{d=1}^{2}
\left(\frac{\widehat y_{bjd}-y_{bjd}}{s_d}\right)^2.
$$

The note verifies gradients and synthetic examples. Important contracts: select valid entries before arithmetic involving potentially invalid values; an all-missing target batch requires skipping the objective/update; subtract large global coordinates at sufficient precision; dead reckoning supplies uncertain labels rather than exact ground truth. The planar frame and axes require verification against real data.

### LLM-A*

The paper modifies search priority using an LLM-generated current waypoint $w$:

$$
f(s)=g(s)+h(s)+d(s,w).
$$

The added distance changes search order, not accumulated edge cost. Waypoints are suggestions, not required path vertices. With the waypoint equal to the goal and matching Euclidean terms, the priority becomes $g+2h$.

The strongest reported Llama 3 few-shot configuration uses 44.59% of A* search operations, 64.02% of search storage, and 102.47% of path length, with 100% benchmark validity. These are reported geometric-mean ratios on the paper's static-map tasks, not runtime, model memory, or driving guarantees. Dynamic weighted A* is an essential comparator: its reported ratios are 60.91% operations and 100.24% path length.

Our pseudocode deliberately differs from the paper:

- Accept exact free graph vertices; reject malformed, nonfinite, off-grid, blocked, and out-of-bounds guidance.
- Remove start/goal and duplicate intermediate guides.
- Advance targets when expanded and skip subsequent already-CLOSED guides.
- Disable waypoint bias when guidance is exhausted.
- Reopen CLOSED states on strict cost improvement.
- Rekey every OPEN entry when the target changes.
- Distinguish resource/deadline exhaustion from proof that no path exists.

The later mathematical appendix derives a conditional bound:

$$
C_{\mathrm{return}}\le(1+\lambda)C^\star,
\qquad 0\le\lambda\le1,
$$

for an appropriately specified metric graph and search. Here $\lambda$ is a proposed dimensionless guide weight, $C^\star$ is optimal graph cost, and $C_{\mathrm{return}}$ is returned path cost. Conditions include positive edge costs dominating geometric distance, exact current priorities, complete successor handling, and correct termination. This is **our derivation**, not the paper's theorem. A separate argument in the appendix also treats no-reopening search under stronger contracts; do not say non-reopening alone disproves this bound.

The existing pseudocode still uses reopening and has not been changed to expose a tunable $\lambda$. Neither bound proves deadline success, low search work, or vehicle feasibility.

### Alpamayo 2 Super

The inspected repository describes a 34B model: 32B VLM plus 2B action expert. Its implemented trajectory path is:

1. Camera/history/prompt encoding and autoregressive reasoning generation.
2. Reuse of the VLM context cache by the action expert.
3. Whole-horizon flow matching over normalized acceleration and curvature.
4. Denormalization and unicycle kinematic rollout.

For normalized clean controls $U\in\mathbb R^{N\times2}$, Gaussian noise $\epsilon$, and dimensionless flow time $\tau$:

$$
\begin{aligned}
Z_\tau&=(1-\tau)\epsilon+\tau U,\\
\mathcal L_{\mathrm{FM}}
&=\mathbb E\left[
\frac{1}{2N}
\left\lVert v_\phi(Z_\tau,\tau;K)-(U-\epsilon)\right\rVert_F^2
\right],\\
Z_{j+1}&=Z_j+\frac{1}{M}v_\phi(Z_j,j/M;K).
\end{aligned}
$$

Here $K$ is the conditioning cache, $v_\phi$ is a learned field in action space, and $M$ is the flow-step count. This field is not physical car velocity or an explicit collision-cost gradient.

Defaults are $M=10$, $N=64$, and physical interval $\Delta t=0.1\,\mathrm{s}$, but loaded checkpoint configuration can override constructor defaults. Ten refinement passes are distinct from a 6.4 s physical prediction horizon.

The physical model uses acceleration $a$, curvature $\kappa$, speed $v$, and yaw $\psi$. Its updates include:

$$
\begin{aligned}
v_{k+1}&=v_k+a_k\Delta t,\\
\psi_{k+1}
&=\psi_k+\kappa_k
\left(v_k\Delta t+\tfrac12a_k\Delta t^2\right).
\end{aligned}
$$

Cartesian position uses trapezoidal integration. Kinematics do not establish tire/actuator feasibility or collision freedom.

Critical source findings:

- The normal generation path does not invoke the separate acceleration/curvature bounds checker or clamp sampled controls. Curvature clipping in training-label preparation is a different operation.
- No graph search or explicit collision checker appears in the inspected trajectory-generation path.
- Returned log-probability values are zeros, not confidence or actual evaluated likelihood.
- Smoke minADE uses recorded future reference positions; it is an offline imitation diagnostic, not online candidate selection or a safety test.
- Separate VLM cross-entropy and expert flow-loss primitives exist. The wrapper does not combine them into a complete training procedure; no complete trainer was found.
- The advanced navigation CFG demo mixes guided/no-navigation fields. The latter retains the guided reasoning text; it is not fully unconditional.
- The standard expert configuration rejects enabled CFG, while the separate demo explicitly overrides sampling with both callbacks.
- Six cameras by four frames describes the validated profile, not arbitrary fisheye support.
- The README's approximately 67/71 GiB on two H100s are reported memory observations, not measured Thor latency.
- The API does not explicitly return other vehicles' predicted trajectories or implement a Leaf tracking controller.

Full equations and file/line references are in the comparison document.

### Latency and efficiency

Let $T_C$ be common pipeline time, $T_A$ ordinary search time, $\rho$ an assumed guided/base search-time ratio, and $T_X$ extra guidance/filter/rekey overhead:

$$
\begin{aligned}
T_B&=T_C+T_A,\\
T_H&=T_C+\rho T_A+T_X,\\
T_H<T_B&\iff T_X<(1-\rho)T_A.
\end{aligned}
$$

Substituting the paper's operation ratio $\rho=0.4459$ is an illustrative assumption, not a conversion validated by measurements. Example: $T_C=20\,\mathrm{ms}$, $T_A=5\,\mathrm{ms}$, and $T_X=0.5\,\mathrm{ms}$ yield $22.7295\,\mathrm{ms}$ versus $25\,\mathrm{ms}$, or 9.082% less time. Against the paper's weighted-A* operation ratio, the same example's extra-overhead allowance is only $0.816\,\mathrm{ms}$.

The proposed 30 ms worksheet allocates 25 ms to named stages and 5 ms to queue/contention/unassigned work. The 20 ms worksheet leaves 3 ms unassigned. Neither explicitly budgets the later guided-search addition; it must consume allowance or replace other work, not be appended to a fully spent budget.

The illustrative transformer core has 37,748,736 parameters and approximately 20.94 GFLOPs at 12 layers, width 512, MLP ratio 4, and 256 total tokens. Vision, embeddings, and other components are excluded. Assumed effective throughput and bandwidth floors are screening arithmetic, not latency predictions.

Fresh-frame throughput differs from service time. At 30 Hz, an arbitrary event can wait almost 33.3 ms for the next exposure even before processing. Report exposure-to-output and event-to-output separately.

## 6. Stuck problems and unresolved choices

There is no tool or permission blocker preventing further documentation. Practical design claims are blocked by missing evidence:

| Unresolved item | Why it matters |
|---|---|
| Dataset path, schema, timestamps, representative samples | No trustworthy input/target pipeline can be specified from “state logs” alone |
| Pose-reference provenance and uncertainty | Logs may lack reliable metric position, heading, or externally validated motion |
| Camera count, placement, resolution, frame rate, exposure, calibration | Determines observability and the sensor/encoder latency budget |
| Structured evidence, decision, reason, and instruction labels | A camera trajectory regressor alone does not establish language grounding or the required reasoning |
| Operating domain | Highway speed, lighting, weather, grade, cut-ins, and other admitted conditions remain unspecified |
| Vehicle geometry and validated dynamic/actuator limits | Generic unicycle defaults are not Nissan Leaf constraints |
| Prediction horizon, model size, graph size/caps, candidate count | Existing values are examples or source defaults, not selected Viktor settings |
| Deadline start/end and acceptance metric | Hard deadline, percentile target, and arbitrary-event response differ |
| Tracker, publication/freshness contract, no-feasible-plan behavior | Geometric paths and predicted poses are not executable vehicle commands |
| Training data volume, cloud budget, simulator access | No measured training schedule or from-scratch feasibility estimate |
| Thor workload measurements | No batch-one, sustained-load, tail-latency, memory, or power measurements |

Do not repeatedly ask for facts already confirmed in Section 3. Only request missing information when it becomes necessary for the next concrete task.

## 7. Recommended next steps

These are continuation options, not work started during this handoff.

1. **Finish the known Markdown formatting cleanup.** In the paper review, replace malformed control words such as `\Verts` and `\vertV`; see Section 8 for exact current locations. Verify rendered math if a renderer is available.
2. **Choose the next narrow experiment from existing notes.** A bounded synthetic ordinary/weighted/guided-A* comparison can proceed independently of vehicle data. A highway training pilot instead needs the dataset audit first. Do not combine both into a new framework.
3. **Audit one representative synchronized recording.** Record schema, units, coordinate conventions, timing, pose provenance, calibration, and available labels. Success means usable input/target contracts without guessed fields.
4. **Freeze timing and output contracts.** Define the clock origin, required structured fields, camera bundle policy, and valid-plan-by-deadline metric. Preserve synchronous reasoning.
5. **Prepare a concrete developer handoff when implementation resumes.** Reuse the existing numerical fixtures and pseudocode contracts. Specify which variant is being implemented and the smallest meaningful checks. The developer should own executable changes; the researcher should own equations and assumptions.
6. **Benchmark fair baselines before expanding architecture.** Compare validity and path quality alongside total overhead, including guidance, graph building, collision checks, rekeying, and failures. For neural alternatives, compare a compact direct head and any flow variant at matched task quality.
7. **Measure the actual deployment path on Thor.** Include acquisition, preprocessing, vision, reasoning, trajectory processing, validation, publication, queues, and thermal/power conditions. Report latency tails, misses, observation age, dropped bundles, and valid-plan coverage.
8. **Only then revise model/training budgets.** Keep random initialization for every learned component. If teacher-generated labels or other external supervision are proposed, document their provenance and distinguish them from loading pretrained weights.

## 8. Pitfalls for the next agent

- **Do not claim code exists because an earlier turn discussed implementation or tests.** The current deliverables are Markdown.
- **Do not silently call an adapted search algorithm a paper reproduction.** Target advancement, reopening, filtering, and exhausted-guide behavior differ.
- **Do not conflate geometric path validity, kinematic consistency, physical feasibility, and closed-loop safety.**
- **Do not turn search-operation ratios into whole-system speedups or search-storage savings into GPU-memory savings.**
- **Do not treat GPU peak sparse throughput as achieved batch-one throughput.** Any weight-streaming lower bound is conditional on the stated memory-traffic model.
- **Do not remove required reasoning from the deadline.** A fast invalid or reason-free output is not a successful cycle.
- **Do not label dead reckoning as exact ground truth, mask invalid labels after NaNs have already entered arithmetic, or report an all-missing batch as zero error.**
- **Do not use zero-filled Alpamayo log probabilities as trajectory confidence.**
- **Do not mistake training-time curvature clipping for an inference-time safety constraint.**
- **Do not mistake flow time for physical driving time or a learned flow field for obstacle-cost descent.**
- **Do not assert a complete Alpamayo training recipe from the available forward/loss methods.**
- **Do not promise 20–30 ms or “not real time” categorically without a workload-specific timing definition and measurements.**
- **Do not overwrite Alpamayo's existing SECURITY.md modification.**
- **Preserve mathematical provenance.** Separate source equations, our derivations, and assumptions; define symbols, dimensions, and units.

### Known remaining formatting issue

Inspection during this handoff found malformed LaTeX in [the paper review](docs/math/llm_astar_paper_review.md), despite earlier delimiter-formatting work:

- Current line 89 uses the token shown as code below instead of a separated norm command and variable.
- Current lines 132–134 join a delimiter command directly to a variable name.

| Existing source token | Intended formatting |
|---|---|
| `\Verts-t\Vert_2` | $\lVert s-t\rVert_2$ |
| `\vertV\vert` | $\lvert V\rvert$ |
| `\vertE\vert` | $\lvert E\rvert$ |
| `\vertO_j\vert` | $\lvert O_j\rvert$ |

These issues were **recorded, not edited**, because the current request is a handoff. Line numbers can move. Delimiter balancing alone does not catch invalid LaTeX command names. The latest Alpamayo comparison was checked separately; do not generalize its passing checks to every older document.

## 9. Collaboration and working style

The user wants mathematical discussion with usable Markdown handoffs and has explicitly corrected unwanted implementation. Be direct, state assumptions, avoid speculative abstractions, make surgical edits, and verify meaningful results.

The researcher role configuration is at /home/barry/Viktor/.codex/agents/researcher.toml. It asks for problem definition, assumptions, notation, derivation, final equations, constraints, pseudocode, checked numerical examples, expected outputs, risks, edge cases, and open questions. Read it when restarting that role.

Use existing documents instead of duplicating specifications. For this project, inline formulas use dollar-delimited math and displays use double-dollar blocks, with aligned or gathered rows for long expressions. Keep complete quantities and units inside math delimiters. Pseudocode identifiers may remain code-formatted.

No commit, push, model download, training, inference, deployment, or vehicle operation was performed for this handoff.

