# Low-latency Viktor: structured reasoning and trajectory generation

Research and source checks: **2026-10-03**. This document records proposals and arithmetic, not measured deployment performance.

## 1. Problem definition

**Confirmed requirement:** the 20–30 ms target includes camera acquisition/processing, current-scene reasoning, and a validated trajectory. Required reasoning outputs are **structured scene evidence, maneuver decision, and reasons**. Moving required reasoning to an asynchronous language process would not satisfy this requirement.

**Confirmed platform/scope:** Jetson AGX Thor 128GB, 273 GB/s published memory bandwidth; AGX Orin 64GB for training, with Thor/cloud training options. First driving tasks are highway lane keeping and vehicle following in Nissan Leaf SV Plus vehicles, using monocular/fisheye cameras. Synchronized recordings/state logs already exist. Every learned vision/language/reasoning/action component starts from random initialization.

**Timing interpretation for analysis:** primary interval is exposure-start/trigger to structured reasoning plus validated-plan output, at most 30 ms, with 20 ms as the stretch allocation. The precise clock origin remains to be agreed. Also report arbitrary-scene-event response, which adds frame-phase waiting. Physical actuator response is a separate measured interval, not silently included or declared fast.

A hard deadline concerns every admitted cycle. A p99 target is a useful engineering gate, but permits misses and does not prove hard real-time behavior. Neither criterion is currently demonstrated; the allocations below are hypotheses.

**Proposed direction:** a compact language-conditioned VLA with fixed-depth synchronous structured reasoning, bounded visual/history tokens, and parallel continuous trajectory candidates. Instructions participate in the same fresh-scene forward pass. No autoregressive prose generation or iterative trajectory denoising is required on this proposed critical path.

**Evidence versus proposal:** NVIDIA describes Alpamayo 2 Super as a policy teacher for subsequent compact-model development. Its public card reports an H100 memory profile, not a millisecond latency result. We cannot convert this into a universal claim that it is “not real time.” [Official model card](https://huggingface.co/nvidia/Alpamayo2-Super), [NVIDIA technical article](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/)

The official implementation generates reasoning text before the diffusion action expert; the model card's profiled setting uses ten diffusion steps. This motivates testing bounded structured outputs and a direct head, but does not establish equal driving quality. [Official repository](https://github.com/NVlabs/alpamayo2), [profile in the official model card](https://huggingface.co/nvidia/Alpamayo2-Super)

## 2. Assumptions

1. Camera timestamps, exposure/readout behavior, transfer, synchronization, and clock domains can be measured. Otherwise “20 ms” has no reproducible start/end definition.
2. The selected exposure/ISP path can preserve useful highway evidence within the sensor budget. If lighting requires longer exposure, smaller neural inference time cannot recover that delay or lost visibility.
3. The operating domain and admitted camera/instruction shapes are bounded. Unknown camera count/resolution prevents a complete front-end compute budget.
4. Compact tokens retain distant lead vehicles, lane boundaries, and relevant changes. Compression is a hypothesis requiring task evaluation; constant token count alone gives no information-preservation guarantee.
5. Fixed-depth structured reasoning suffices for the first tasks. If it cannot produce grounded decisions, extra synchronous depth must be budgeted rather than replaced by off-path explanations.
6. Required lane/lead/ego information and uncertainty can support candidate validation. Existing state logs do not imply these labels or online estimates already exist.
7. Resource-rate estimates describe the actual precision, batch-one kernels, power state, and concurrent work. Peak sparse TOPS cannot substitute for them.
8. A bounded latest-observation scheduler and a separately defined tracking/fallback interface exist before closed-loop deployment. Otherwise queue age or an empty feasible set has undefined operational behavior.
9. The initial planar trajectory representation remains appropriate only for the audited domain; the existing mathematical handoff's frame/reference assumptions still apply.

## 3. Notation

Role labels: I=input, O=output, C=design constant, P=learned parameter, V=derived variable.

| Symbol | Meaning / role | Shape/domain | Units |
|---|---|---|---|
| $t_e,t_0,t_r,t_p,t_a$ | Event onset, exposure/trigger start, frame-bundle readiness, validated output, actuator effect / V | Scalar timestamps | s |
| $t_{c,i}$ | Exposure-start timestamp of current contributing camera $i$ / I | $C$ finite timestamps | s |
| $D$ | Proposed total deadline / C | Positive scalar; examples 0.020 or 0.030 | s |
| $T_{\rm cam},T_q,T_v,T_f,T_h,T_g,T_o$ | Sensor path, queue, lens remap/crop/normalization plus vision, fusion/reasoning, trajectory head, validation, output durations / V | Nonnegative scalars | s |
| $T_{\rm all},T_{\rm ready},T_{\rm event},T_{\rm act}$ | Full admitted cycle, ready-to-plan, event-to-plan, plan-to-actuation durations / V | Nonnegative scalars | s |
| $W,f_{\rm cam},f_{\rm out}$ | Event/frame-phase wait, camera rate, fresh-plan throughput / V/C | $W\ge0$; positive rates | s, Hz, Hz |
| $b_i,\epsilon_i$ | Stage time allowance and probability of exceeding it / C | $b_i\ge0,\;0\le\epsilon_i\le1$ | s, dimensionless |
| $A,A_{\rm scene},A_{\rm mem},A_{\max}$ | Age argument, current scene age, memory age, validated age limit / V/C | Nonnegative scalars | s |
| $v,v_{\max},a_{\max},\delta_0,\delta(A)$ | Illustrative speed; initial relative-speed bound; subsequent relative-acceleration bound; initial position-error bound; actual relative-position deviation from stored reference / I/C/V | Nonnegative scalars | m/s, m/s, m/s², m, m |
| $C,H,W_{\rm img}$ | Camera count and image height/width / C | Positive integers; unknown | dimensionless counts |
| $I,e,\ell,\mathcal C$ | Current images, ego state, instruction tokens, calibration / I | Images $[C,3,H,W_{\rm img}]$; other schemas pending | Normalized pixels/token IDs dimensionless; state/calibration SI units audited |
| $n_v,n_m,n_\ell,n_e,n_r,M$ | Visual, memory, instruction, ego, reason-query, mode-query token counts / C | Bounded nonnegative counts; active streams positive | dimensionless |
| $N,d,L,r$ | Total tokens, hidden width, layers, plain FFN expansion / C | Positive integers | dimensionless |
| $Z,\mu,\theta$ | Fused tokens, bounded memory state, learned parameters / V/I/P | $Z:[N,d]$, $\mu:[n_m,d]$, parameters architecture-dependent | Dimensionless normalized computational quantities |
| $\mathcal R$ | Explicit structured evidence/decision/reasons/uncertainty / O | Fixed maximum slot schema; defined below | Attribute units declared by field; probabilities dimensionless |
| $\tau_m,K,D_\tau,\Delta_j$ | Candidate, sample count, coordinate width, future offsets / O/C | $\tau:[M,K,D_\tau]$; increasing positive offsets | For initial $D_\tau=2$, positions m; offsets s |
| $z_m,g_j,\mathcal F,J,m^\star$ | Mode score, normalized constraint residuals, feasible indices, constraint count, selection / O/V/C | Scalars; set $\mathcal F\subseteq\{1,\ldots,M\}$; $J\ge1$ | Scores/residuals/indices dimensionless |
| $P_{\rm core},P_{\rm head},P_{\rm model}$ | Core/head/total learned parameter counts / V | Nonnegative integers | parameter count |
| $F,F_{\rm core},F_{\rm head}$ | Floating-point operation counts / V | Nonnegative scalars; multiply-add counted as two | FLOP |
| $Q,b_w,BW_{\rm eff},P_{\rm eff}$ | Transferred bytes, bytes/weight, effective bandwidth, effective compute rate / V/C | Nonnegative; rates positive | byte, byte/parameter, byte/s, FLOP/s |
| $S,J_{\rm txt},T_{\rm step},T_{{\rm tok},j},T_{\rm shared},T_{\rm pre}$ | Diffusion steps, generated text tokens, respective pass times, shared encoding, text prefill / C/V | Counts nonnegative; times nonnegative | dimensionless counts; s |

“Unknown,” invalid evidence, absent object, and padded slot are distinct schema states. No camera layout, motion-reference quality, label inventory, or trained vocabulary is inferred from this notation.

## 4. Derivation

**A. Measure the complete critical path.**

1. Define $T_{\rm cam}=t_r-t_0$ to include the actual exposure/readout/ISP/transfer/synchronization path. These operations may overlap; measure the sensor dependency path rather than adding overlapping stopwatch intervals.
2. After the required bundle is ready, account for queue, vision, fusion/reasoning, candidate generation, validation, and publication. Vision time $T_v$ explicitly includes lens remapping, cropping, normalization, encoding, and pooling. For the proposed serial dependency path, their sum gives $T_{\rm ready}$; overlapping implementations require their dependency-path duration instead.
3. Thus $T_{\rm all}=T_{\rm cam}+T_{\rm ready}$. Arbitrary-event response adds $W=t_0-t_e$ when the event must wait for the next usable exposure.
4. A periodic ideal camera has $0\le W<1/f_{\rm cam}$. This is a frame-phase bound only; missed triggers, buffering, or exposures that fail to reveal the event can make actual observation delay larger.
5. Throughput counts fresh completed observations per second. A 20 ms neural service time does not create 50 Hz fresh observations from 30 Hz cameras.

**B. Replace variable-length generation with a bounded reasoning path.**

1. A shared lightweight image encoder processes fresh calibrated views. Pool to a fixed total $n_v$ visual tokens; this is total across views, not a hidden per-camera multiplier.
2. Concatenate these with bounded history memory, bounded instruction tokens, ego state, reason queries, and mode queries. A small randomly initialized multimodal transformer contextualizes all of them in the current cycle.
3. Fixed-slot heads produce $\mathcal R$: scene evidence references/attributes, maneuver distribution, reason relations, and validity/uncertainty. Multiple causes are allowed up to the declared slot limit.
4. A small trajectory head consumes the same explicit structured probabilities/attributes that are reported in $\mathcal R$, plus fused context and mode queries. Consequently reasoning precedes and conditions the trajectory inside one fixed-depth network evaluation; it is not an explanation attached afterward.
5. All $M$ trajectories and scores are produced in parallel. A direct waypoint head is the first latency ablation, consistent with the existing pilot; physical feasibility remains a separate gate. Control-space output/integration is an alternative whose latency and dynamics assumptions must be included.
6. Optional prose/templates may improve readability, but the required structured result already appears synchronously. No equivalence to open-ended natural-language chain-of-thought is claimed.

**C. Count a proposed standard dense core, not Alpamayo's architecture.**

For full self-attention with width $d$, ordinary Q/K/V/output projections require $4d^2$ weights. A two-projection FFN of width $rd$ requires $2rd^2$. Biases, normalization, embeddings, vision/pooling, reasoning heads, and other adapters are excluded.

For $N$ tokens, projection work is $8Nd^2$ FLOPs and FFN work $4rNd^2$. The attention score and value products together require $4N^2d$ FLOPs. Sum across $L$ layers. This is a cost approximation, not an accuracy or runtime model.

A shared linear head mapping each mode query to $KD_\tau$ coordinates and one score requires approximately $d(KD_\tau+1)$ weights and $2Md(KD_\tau+1)$ FLOPs. Additional conditioning/reasoning layers are extra.

**D. Separate resources from sequential dependencies.**

Computation requires at least $F/P_{\rm eff}$ seconds, and memory traffic at least $Q/BW_{\rm eff}$, if the chosen rates are valid attainable ceilings for the workload. With perfect overlap the larger remains a conditional lower bound. Launch, synchronization, contention, and serial stages can only increase it.

An iterative action expert performs $S$ dependent updates; producing candidates in a batch does not remove this sequential step count. Autoregressive explanation generation similarly incurs $J_{\rm txt}$ dependent token steps. The proposed structured head removes those loops, not the need to demonstrate adequate reasoning.

## 5. Final equations

$$
\begin{aligned}
T_{\rm ready} &= T_q+T_v+T_f+T_h+T_g+T_o, \\
T_{\rm all} &= T_{\rm cam}+T_{\rm ready}=t_p-t_0.
\end{aligned}
\tag{1}
$$

$$
\begin{aligned}
T_{\rm event} &= W+T_{\rm all}, \\
T_{\rm act} &= t_a-t_p, \\
A_{\rm scene} &= t_p-\min_i t_{c,i}.
\end{aligned}
\tag{2}
$$

$$
\begin{gathered}
\sum_i b_i\le D,\qquad
\Pr(T_i>b_i)\le\epsilon_i,\qquad
\sum_i\epsilon_i\le0.01 \\[4pt]
\Longrightarrow\quad \Pr(T_{\rm all}>D)\le0.01.
\end{gathered}
\tag{3}
$$

Equation (3) uses the union bound and nonnegative additive stages; independence is unnecessary. Adding individual stage p99 values alone does **not** establish total p99. A hard guarantee instead requires defensible bounds for every admitted execution.

$$
\begin{aligned}
N &= n_v+n_m+n_\ell+n_e+n_r+M, \\
P_{\rm core} &\simeq L(4+2r)d^2, \\
F_{\rm core} &\simeq L\left[(8+4r)Nd^2+4N^2d\right].
\end{aligned}
\tag{4}
$$

With $r=4$, these reduce to $12Ld^2$ parameters and $L(24Nd^2+4N^2d)$ FLOPs.

$$
\begin{gathered}
T_{\rm device}\ge\max\left(\frac{F}{P_{\rm eff}},\frac{Q}{BW_{\rm eff}}\right), \\[4pt]
Q_{\rm weights}=b_wP_{\rm model}
\quad\text{for a complete single weight stream}.
\end{gathered}
\tag{5}
$$

Here $T_{\rm device}$ denotes the device portion being modeled, in seconds. Equation (5) excludes camera time and assumes specified resource ceilings; a weight-only calculation omits activations, reloads, quantization metadata, and other concurrent traffic.

$$
\begin{aligned}
T_{\rm diffusion} &= T_{\rm shared}+\sum_{s=1}^{S}T_{{\rm step},s}, \\
T_{\rm text} &= T_{\rm pre}+\sum_{j=1}^{J_{\rm txt}}T_{{\rm tok},j}.
\end{aligned}
\tag{6}
$$

The subscript $s$ indexes successive diffusion passes. Neither sum can be replaced by one pass merely because GPU arithmetic is parallel.

$$
\begin{gathered}
\mathcal F=\left\{m:
\begin{array}{l}
\mathcal R\text{ has required valid evidence and} \\
g_j(\tau_m,e,\mathcal R)\le0\quad\forall j\le J
\end{array}
\right\}, \\[6pt]
m^\star=\arg\max_{m\in\mathcal F}z_m.
\end{gathered}
\tag{7}
$$

Define a deterministic tie rule. If $\mathcal F=\varnothing$, return no validated trajectory and a reason to the separately specified fallback/tracker interface. Never return an unchecked highest-score candidate.

$$
\delta(A)\le\delta_0+v_{\max}A+\frac{1}{2}a_{\max}A^2,
\qquad 0\le A\le A_{\max}.
\tag{8}
$$

Equation (8) bounds actual relative-position deviation from a stored estimate: initial error is at most $\delta_0$, initial relative speed at most $v_{\max}$, and relative acceleration throughout the interval at most $a_{\max}$. Integrating velocity and applying the triangle inequality gives the bound in metres. It is not a visibility guarantee; it cannot bound an entirely unobserved new hazard or compensate for invalid motion bounds.

## 6. Constraints

**Structured reasoning contract:** bounded evidence slots identify an observed region/object reference, relevant attribute, observation time, and validity. Maneuver output covers lane keeping plus longitudinal following/slow/hold intent within the first task. Reason relations explicitly connect evidence to intent; uncertainty and “insufficient evidence” are output states. Exact labels require an annotation specification, not invention from the existing logs.

The trajectory head consumes the reported distributions/numerical attributes, rather than a hidden alternative decision. Readable argmax labels are presentation choices; distribution/uncertainty must remain accessible. This improves inspectability but does not prove causal faithfulness.

**Provisional design search, not committed configuration:**

| Quantity | Initial hypothesis | Condition for retaining it |
|---|---|---|
| Fusion depth/width | $L=8$–12, $d=384$–768, ordinary FFN $r=4$ | Held-out task quality and measured deadline |
| Visual tokens | $n_v=128$–256 total | Small/distant evidence survives compression |
| Memory/instructions | $n_m=32$–96; $n_\ell=8$–24 | Temporal observability and accepted instruction coverage |
| Reason/mode queries | $n_r=4$–12; $M=2$–4 | Sufficient causes and distinct useful plans |
| Total parameters | Not fixed; example core range 14.16–84.93M | Add vision, embeddings, heads, and adapters before sizing |

These ranges are our hypotheses, not measurements or requirements from Alpamayo. Camera count/resolution can dominate even when $n_v$ is fixed. The total vocabulary/embedding size and model-language capability remain to be established.

No silent instruction truncation: reject or explicitly handle inputs outside the admitted token budget. Process instructions synchronously in the compact core; cache unchanged per-frame-independent features only as an optional optimization. Cache misses, new instructions, memory resets, and cold-start cycles must have declared timing behavior.

Memory is bounded and timestamped. Reusing a per-frame visual feature is different from reusing a bidirectional transformer's KV cache after its window changes; the latter is invalid without an architecture designed for such reuse. Current scene reasoning is recomputed every cycle.

Age limits, uncertainty margins, lane/lead validation, inter-sample collision checks, and vehicle dynamics limits require measurement/labels. An unknown constraint is not satisfied merely because its score is missing. Waypoint bounds alone do not establish continuous-time feasibility.

Thor's 273 GB/s is a published bandwidth, not reserved capacity for this model; CPU/GPU sensor, tracker, display, and explanation workloads can contend for shared memory. Priority labels or asynchronous launch do not establish isolation. [Thor specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)

Static-shape execution/CUDA graphs can reduce launch overhead, but do not erase arithmetic, memory traffic, or sensor time. They are secondary after the workload fits the mathematical budget. [TensorRT optimization guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)

## 7. Algorithm / pseudocode

1. Timestamp acquisition through synchronized-bundle readiness; preserve per-camera exposure timestamps and capture quality.
2. Admit only the newest suitable bundle and an explicitly versioned instruction. Bound backlog; record skipped/stale bundles instead of quietly processing FIFO history.
3. Encode fresh images; combine compact visual tokens, bounded aged memory, instruction tokens, and ego state.
4. Run fixed-depth multimodal fusion and synchronous structured evidence/decision/reason heads.
5. Condition the parallel trajectory head on the reported structure; produce $M$ candidates and their scores.
6. Validate candidates with audited constraints and uncertainty; select only from $\mathcal F$. Emit structured missing-evidence/no-feasible-plan results when applicable.
7. Publish the plan and reasoning with observation/output timestamps; update bounded memory. Record total latency, age, and rejected/deadline-missed cycles.

Fusion complexity follows Eq. (4); the direct linear trajectory head is $O(MdKD_\tau)$. Candidate checking grows with $M,K$ and the scene/constraint representation. Reasoning schema, validator, input transfer, and memory update costs must be included, not treated as free CPU work.

**Training/research gates:** start all learned weights randomly; establish the direct-trajectory baseline, add supervised evidence/decision/reason slots and language grounding, then test the coupled model. Existing video/state data do not establish language or causal-reason labels. Compare with/without structured conditioning on matched data; test whether conflicting/missing evidence changes both reasoning and plan appropriately.

Only retain the bounded architecture if closed-loop lane/following quality, evidence correctness, reasoning–action consistency, and timing all meet declared gates. A larger/faster GPU does not replace missing supervision. No pretrained teacher is assumed.

**Representative measurement matrix:** actual camera modes and exposure conditions; cold/warm runs; instruction changes; stopped/fast traffic; memory reset/maximum memory; invalid evidence; full tracker/sensor concurrency; sustained power/thermal conditions. Report exposure-to-output and ready-to-output distributions, p99 and higher tails/max observed, miss rates, fresh-plan throughput, current-scene age, and exclusions. Count valid-plan-by-deadline coverage over every admitted cycle; fast invalid/no-feasible-plan outputs are not successful validated-plan completions. Separately report captured/admitted/dropped bundles and all deadline misses so admission cannot hide poor coverage. Maximum observed latency is not a proven worst-case bound.

## 8. Example calculations

All numerical examples below are arithmetic design screens, not measured Viktor latency or selected vehicle limits.

**Whole-cycle allocation worksheet**

| Critical-path component | 30 ms allocation | 20 ms stretch allocation |
|---|---:|---:|
| Acquisition/readout/ISP/transfer/synchronization | 6 ms | 4 ms |
| Lens remap/crop/normalization, visual encoder, pooling | 8 ms | 5 ms |
| Multimodal fusion and structured reasoning | 6 ms | 4 ms |
| Parallel trajectory generation | 1 ms | 1 ms |
| Candidate validation | 3 ms | 2 ms |
| Result publication | 1 ms | 1 ms |
| Queue/contention/unassigned allowance | 5 ms | 3 ms |
| Total | 30 ms | 20 ms |

The camera row is unverified and may invalidate either allocation. These are per-cycle allowances, not independently measured p99 numbers. Seven equal risk allocations under Eq. (3) would be $0.01/7\approx0.00142857$ each; no empirical tail probabilities have been established.

**Compact-core calculation:** let $L=12,d=512,r=4,N=256$, with illustrative token decomposition $160+64+16+4+8+4$. Equation (4) gives $P_{\rm core}=12\cdot12\cdot512^2=37,748,736$, and $F_{\rm core}=12(24\cdot256\cdot512^2+4\cdot256^2\cdot512)=20,937,965,568$ FLOPs.

At a hypothetical effective 10 TFLOP/s, this core's compute floor is 2.0937965568 ms. Its FP16 weights alone stream in $2P_{\rm core}/(273\times10^9)=0.2765475165$ ms at peak published bandwidth. The larger conditional floor is 2.0938 ms; none of the excluded components or execution overhead is thereby solved.

For $M=4,K=32,D_\tau=2$, the shared linear head has 33,280 weights and 266,240 FLOPs. This sample count is only an arithmetic fixture, not a committed horizon. Integration, structured conditioning, scoring calibration, and validation remain extra.

Increasing $N$ with the same core gives 10.0663296 GFLOPs at 128 tokens, 45.097156608 at 512, and 257.69803776 at 2048. At the same hypothetical rate, 2048 tokens consume a 25.7698 ms compute floor before cameras and other work.

**Large-model bandwidth screen:** a full single stream of a nominal 32B backbone at 2 byte/weight is 64 GB, requiring 234.4322 ms at 273 GB/s. Even ideal packed 4-bit weights require 16 GB and 58.6081 ms. This is conditional on streaming those weights, excludes metadata/activations/action expert, and is not an Alpamayo latency measurement. Its architecture differs from Eq. (4). [Published model configuration](https://huggingface.co/nvidia/Alpamayo2-Super/raw/main/config.json)

For comparison, one FP16 stream of hypothetical 100M/200M/300M models has bandwidth floors 0.7326/1.4652/2.1978 ms. These figures neither establish training feasibility nor prove inference deadlines.

**Observation timing:** an arbitrary event can wait almost 33.3333/16.6667/8.3333 ms for ideal 30/60/120 Hz sampling. A 20 ms exposure-to-output cycle then approaches 53.3333/36.6667/28.3333 ms event-to-output. Therefore 30 Hz is compatible with a 20 ms exposure-based cycle, but cannot guarantee a 30 ms arbitrary-event deadline even before other delay.

At an illustrative 30 m/s, 20 and 30 ms correspond to 0.6 and 0.9 m traveled. Adding a 30 Hz phase wait to the 30 ms cycle gives up to about 1.9 m before actuator effects. This is delay distance, not a braking-distance or safety-margin calculation.

**Selection:** with scores $(0.7,0.9,0.4)$ and feasibility $(1,0,1)$, Eq. (7) selects candidate 1. If feasibility becomes $(0,0,0)$, there is no validated candidate, irrespective of score.

## 9. Expected outputs

Arithmetic was checked in a shell Python session; no executable implementation or verification files are part of this handoff. Integer parameter/FLOP results are exact. Decimal time checks agree within $10^{-6}$ ms of the rounded values below.

| Check | Verified result |
|---|---:|
| Core parameter count | 37,748,736 |
| Core FLOPs, $N=256$ | 20,937,965,568 |
| Core compute time at hypothetical 10 TFLOP/s | 2.093797 ms |
| Core-only FP16 stream at 273 GB/s | 0.276548 ms |
| Direct-head weights / FLOPs | 33,280 / 266,240 |
| Core-range endpoints from $L,d$ hypotheses | 14,155,776 / 84,934,656 parameters |
| 32B single stream, 2 byte / 0.5 byte | 234.432234 / 58.608059 ms |
| 100M / 200M / 300M FP16 stream | 0.732601 / 1.465201 / 2.197802 ms |
| 30 ms / 20 ms worksheet sums | 30 / 20 ms |
| 20 ms cycle plus 30 / 60 / 120 Hz phase bounds | 53.333333 / 36.666667 / 28.333333 ms |
| Score selection with feasibility $(1,0,1)$ | Candidate 1 |
| Score selection with empty feasible set | No validated plan |

No data-driven fit, GPU benchmark, statistical confidence interval, or causal-accuracy result has been obtained.

## 10. Potential mathematical risks

Resource floors can be very optimistic: useful precision/kernel throughput, activation traffic, memory residency, launch overhead, and concurrent sensor work are unresolved. Sparse FP4/FP8 advertised rates cannot be inserted as dense $P_{\rm eff}$. A theoretical fit is necessary, not sufficient.

Pooling may erase small/distant hazards; bounded memory may miss interactions; shorter exposure may weaken visual evidence. The structured bottleneck can produce plausible but false reasons, or omit a cause outside its slot inventory. Direct candidates can collapse to duplicates or average incompatible futures. Faster output does not validate the semantics.

Constraint checking over predicted scene state inherits scene uncertainty. Finite samples can miss violations between waypoints. Age bounds do not reveal occluded/new objects. When required evidence or a trustworthy margin is absent, the gate must expose that uncertainty.

Old memory is historical context, not current evidence. Ego-frame changes require consistent motion/time handling. Caching contextual KV tensors across changing bidirectional windows can silently corrupt state; static-instruction caching cannot justify stale scene reasoning.

An asynchronous explanation process sharing Thor can still delay the synchronous path. Reordered queues, dropped bundles, or batching can improve throughput while worsening observation age. Report all three separately.

Numerical checks must reject nonfinite scores/coordinates and invalid scales/timestamps. Stable categorical normalization subtracts the maximum logit before exponentiation; “unknown” is an explicit category, not a NaN. Ties and empty feasible sets need deterministic behavior. This specification claims neither convex training nor convergence nor hard real-time certification.

## 11. Results in limiting / edge cases

- Zero frame-phase wait: event-to-output equals exposure-to-output. Infinite camera rate removes phase wait only, not exposure/readout or computation.
- Zero required text tokens is not an acceptable substitute for language conditioning; unsupported/missing instructions need the declared task policy, not silent semantic invention.
- Zero memory tokens recovers a current-frame ablation. Stationary context may make reuse useful, but does not establish freshness for moving objects.
- $N$ large: the $N^2d$ term dominates. Small $N$: projection/FFN work and launch/memory overhead can dominate.
- $S=1$ removes repeated diffusion steps but does not prove a single diffusion evaluation matches a trained direct decoder. $J_{\rm txt}=0$ removes generated prose, while synchronous $\mathcal R$ remains mandatory.
- $M=1$ becomes a single-candidate ablation; no-feasible-plan handling is still required. Increasing $M$ increases head/check work and does not guarantee coverage of a safe mode.
- $A=0$ reduces Eq. (8) to the initial uncertainty. Zero motion bounds keep that allowance fixed; unbounded/invalid motion prevents a finite age-based claim.
- Empty/invalid camera bundles, expired memory needed for a decision, no evidence, and empty feasible sets emit explicit invalidity rather than an invented valid plan.
- A p99 success with nonzero misses is not a hard-deadline success. A hard event deadline below one full sampling period is impossible under the ideal periodic worst-phase assumption.

**Open questions for the user**

1. Should the strict 20–30 ms deadline begin at exposure/trigger, or at an arbitrary new scene event? Is 30 ms a hard bound or a percentile target with a defined miss policy?
2. What camera count, positions, resolutions, frame rates, exposure limits, trigger/synchronization, and transport/ISP path will be admitted?
3. What highway speed/visibility/road conditions define the first domain, and what tracking/other GPU workloads share Thor?
4. Which available data can support evidence/reason labels, language conditioning, metric trajectories, uncertainty, and candidate-validation metrics?
