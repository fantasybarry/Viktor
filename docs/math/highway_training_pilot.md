# Highway training pilot: preliminary mathematical handoff

**Scope:** mathematical research and Markdown documentation. Implementation is assigned to other agents by the user. This handoff records proposed equations, assumptions, numerical requirements, and verified examples for those agents.

## 1. Problem definition

**Confirmed:** Viktor's first milestone is highway lane keeping and vehicle following in 2025 Nissan Leaf SV Plus EVs. Vision uses monocular/fisheye cameras. AGX Orin 64GB is available for training; Thor is the deployment target and an optional training resource; rented GPUs are an option. Synchronized camera recordings and vehicle-state logs exist. All learned model components must start from random initialization.

**Unresolved:** dataset location/schema, camera layout/calibration, state fields, usable motion reference, language annotations, operating conditions, prediction horizon, architecture size, and training duration. Existing state logs do not imply measured poses, lead-vehicle tracks, or instruction labels.

**Proposed diagnostic:** predict future ego-frame positions from calibrated camera history and audited ego state, then measure imitation error. This is an ablation within the from-scratch VLA effort, not a complete LLM, language-grounding solution, vehicle controller, or proof of safe following. Do not add lane changes to this milestone.

Alpamayo-R1 §3, Eqs. (1)–(3), combines observations, reasoning, and trajectories; §5.1 uses discrete action supervision and a separate continuous decoder. Its pretrained backbone and numerical horizon are not requirements for this pilot. The rigid transform and normalized regression below are our proposed baseline, not copied Alpamayo objectives.

Source: [Alpamayo-R1 v2, local paper](/home/barry/Downloads/2511.00088v2.pdf).

## 2. Assumptions

1. Audited timestamps align images, state, and future reference within a measured tolerance. Otherwise input/target misalignment can imitate reaction delay.
2. Reference positions and anchor heading share a calibrated, metric frame. Otherwise the target transform is invalid. Sensor estimates remain uncertain references, not exact ground truth.
3. A local planar frame adequately describes the selected clips. Significant grade, banking, or roll/pitch requires a justified 3D transform/projection; the planar derivation must then be revised.
4. The proposed ego axes are forward/left, with heading positive counterclockwise about up. Other source conventions require explicit conversion; guessing flips lateral targets.
5. Future samples have positive, increasing time offsets and an approved reconstruction rule. Missing/poor references are masked; extrapolation across missing intervals is not assumed valid.
6. Each accepted input has adequate camera/state history under a documented completeness policy. The pilot does not silently replace missing input sensors with zeros.
7. Fixed positive position scales are chosen using training data or declared task tolerances, without validation/test leakage. These scales change objective weighting; their values remain unset.
8. A deterministic trajectory is an intentional diagnostic simplification. Ambiguous scenes can make squared-error averages poor actions; passing this baseline does not settle the final decoder.

## 3. Notation

All shapes below describe the proposed mathematical interface, not the uninspected log schema.

| Symbol | Meaning / role | Shape | Domain and units |
|---|---|---|---|
| $B,T,C,H,W,K,d_e$ | Batch, history, cameras, image height/width, future samples, state width; design constants | Integers | Positive; dimensionless counts |
| $b,i,c,j,d$ | Batch/history/camera/future/coordinate indices; variables | Scalars | Respective valid integer ranges; $d\in\{1,2\}$, dimensionless |
| $I$ | Audited, normalized camera history; input | $[B,T,C,3,H,W]$ | Finite dimensionless values; common RGB grid is a proposed adapter contract |
| $e$ | Audited ego-state history; input | $[B,T,d_e]$ | Fields/ranges/SI units must be enumerated after schema inspection |
| $\mathcal C$ | Camera projection, intrinsics, extrinsics, time conventions; input calibration | $C$ calibration records | Structure/units depend on audited lens model; not inferred |
| $t_{bi},t_{b0},\Delta_j$ | Historical times, anchor time, future offsets; inputs/constants | $[B,T],[B],[K]$ | Seconds; $t_{bi}\le t_{b0}$, $0<\Delta_1<\cdots<\Delta_K$ |
| $p^W_{b0},p^W_{bj}$ | Anchor/future position references; inputs | $[B,2],[B,K,2]$ | Metres; anchor/valid targets finite in $\mathbb R^2$; future time $t_{b0}+\Delta_j$ |
| $\psi_{b0}$ | Anchor heading reference; input | $[B]$ | Finite real angle, radians |
| $R(\psi),u_{bj}$ | Ego-to-world planar rotation and relative world displacement; derived variables | $[2,2],[2]$ | Rotation dimensionless; displacement metres |
| $y_{bj},\hat y_{bj}$ | Local target and predicted future position; target/output | Each $[2]$; batched $[B,K,2]$ | Metres; valid entries finite |
| $f_\theta,\theta$ | Predictor and all learned parameters | Architecture-dependent | Maps declared inputs to metres; finite real parameters, units follow the future architecture/adapters; random initialization |
| $m_{bj},\mathcal V,N$ | Target validity, valid index set, its size; data/derived variables | $[B,K]$, set, scalar | $m\in\{0,1\}$; $N$ nonnegative count, dimensionless |
| $s_d,r_{bjd},L$ | Position scale, normalized residual, objective; constant/derived variables | $[2]$, scalar, scalar | $s_d>0$ metres; $r,L$ dimensionless |

## 4. Derivation

1. Audit whether metric positions/heading are available or reconstructible. Positions integrated from speed/heading are **weak labels** with accumulated error; document their origin, uncertainty, and validation separately from externally referenced poses. Do not rename dead reckoning as ground truth.
2. For each accepted anchor, subtract its world position: $u_{bj}=p^W_{bj}-p^W_{b0}$. Translation places the origin at the anchor; both terms use metres.
3. Coordinate conversion from ego to world is $u_{bj}=R(\psi_{b0})y_{bj}$. Since a rotation is orthogonal, $R^\top R=I_2$, multiplication by $R^\top$ gives $y_{bj}=R^\top u_{bj}$. Here $I_2$ is the dimensionless $2\times2$ identity.
4. Retain only valid reference pairs: $\mathcal V=\{(b,j):m_{bj}=1\}$, $N=\vert\mathcal V\vert$. Validity follows reference availability/quality rules, never the model's error or predictions.
5. Divide each position error by its declared scale: $r_{bjd}=(\hat y_{bjd}-y_{bjd})/s_d$. Metres cancel. Average squared residuals over the $2N$ valid scalar coordinates.
6. If $N=0$, no supervised objective exists for that batch. Skip its optimizer update and report coverage; returning an artificial zero would misreport an unevaluated batch as perfect.
7. For $N>0$, differentiating a valid squared residual gives $2(\hat y-y)/s_d^2$; multiplying by $1/(2N)$ cancels the 2. Invalid predictions do not occur in the sum, so their derivatives are zero. These are derivatives with respect to prediction entries, not a guarantee about upstream operations that themselves generate NaNs.

## 5. Final equations

$$
\begin{aligned}
R(\psi) &=
\begin{bmatrix}
\cos\psi & -\sin\psi \\
\sin\psi & \cos\psi
\end{bmatrix}, \\[4pt]
y_{bj} &= R(\psi_{b0})^\top\left(p^W_{bj}-p^W_{b0}\right).
\end{aligned}
\tag{1}
$$

$$
\begin{aligned}
\hat y &= f_\theta\left(I,e,\mathcal C,\{t_{bi}-t_{b0}\},\{\Delta_j\}\right), \\[4pt]
L &= \frac{1}{2N}\sum_{(b,j)\in\mathcal V}\sum_{d=1}^{2}
\left(\frac{\hat y_{bjd}-y_{bjd}}{s_d}\right)^2,
\qquad N>0.
\end{aligned}
\tag{2}
$$

For $N=0$: **SKIP**, with no gradient/update. Equation (1) preserves metres; Eq. (2)'s objective is dimensionless. Gather valid targets/predictions before arithmetic: multiplying a NaN residual by zero is not a valid masking implementation. Use adequate precision for large world-coordinate subtraction and finite residual accumulation.

$$
\frac{\partial L}{\partial\hat y_{bjd}} =
\begin{cases}
\dfrac{\hat y_{bjd}-y_{bjd}}{N s_d^2}, & (b,j)\in\mathcal V, \\[6pt]
0, & (b,j)\notin\mathcal V,
\end{cases}
\qquad N>0.
\tag{3}
$$

The gradient has units $\mathrm{m}^{-1}$. There is no loss/gradient when $N=0$, rather than an invented zero-loss tensor.

**Numerical requirements for a future implementation:** coordinate inputs have shapes $[B,K,2],[B,2],[B]$; output shape is $[B,K,2]$. The loss takes prediction/target arrays of shape $[B,K,2]$, a validity mask $[B,K]$, and fixed positive scales $[2]$. The derivative in Eq. (3) must be preserved when integrating the objective into training.

Coordinate conversion requires finite positions, anchors, and headings. Missing-reference selection and mask construction belong to label preparation. Only the loss excludes invalid pairs; active values and the computed objective must be finite. Select valid entries before arithmetic to prevent undefined values from contaminating the result.

At empty batch or horizon boundaries, the transformed result is empty and no supervised loss or optimizer update is defined. Positive finite scales remain required. Preserve adequate precision when subtracting large world coordinates: converting already-rounded float32 coordinates to float64 cannot restore lost offsets. The position differences can be converted to a lower precision afterward if its error is acceptable. Low-precision training requires separate analysis of residual, loss, and gradient overflow and underflow.

## 6. Constraints

- Inspect sample files first: timestamps/clock domains, camera IDs/lens calibration, image geometry, state units/frames, pose provenance, discontinuities, and missingness. No parser or log-field names are specified here.
- A validity mask covers the whole 2D target. If only one coordinate is trustworthy, this baseline rejects the point. A missing mask requires constructing one under an audited rule; it does not mean all labels are valid.
- Reject malformed shapes, nonpositive/nonfinite scales, or inadequate input context. Nonfinite **valid** targets or predictions are errors to investigate, not additional silent mask entries.
- Reference reconstruction, resampling, interpolation tolerances, and maximum trustworthy horizon require evidence. No pose reference means Eq. (1) is unavailable until a validated label-generation route exists.
- Eq. (2) enforces neither dynamics nor collision avoidance. Leaf acceleration, steering, braking, jerk, and delay bounds remain unmeasured; do not fit or invent constants from the trim name.
- Architecture and batch/token budget must follow representative training-memory/throughput and Thor latency measurements. Hardware availability alone supplies no training-duration estimate.

## 7. Algorithm / pseudocode

1. Audit representative data and document units, calibration, reference provenance, and validity rules.
2. Split complete drives/routes before windowing; keep overlapping temporal windows out of different splits.
3. Select an anchor, audited camera/state history, and candidate future offsets; require sufficient input context.
4. Construct targets only where reference support exists; construct validity independently of predictions.
5. Predict positions with a randomly initialized candidate; gather valid pairs and compute Eq. (2), or SKIP if none.
6. Check loss/gradients for finiteness, update on valid supervision, and report valid-point/sample coverage alongside loss.
7. Compare held-out errors and representative closed-loop behavior; benchmark full training steps and Thor end-to-end latency before selecting size.

Target conversion/loss require $O(BK)$ work and $O(BK)$ storage, excluding input preparation and the unchosen network. Network complexity and memory must be measured for its actual shapes.

Closed-loop evaluation must include lateral/heading errors, headway violations, collisions, braking-response delay, and p99 sensor-to-plan latency, with scenario counts and exclusions. Report motion error in metres alongside normalized loss. Lane/lead geometry and event ground truth needed for these metrics are separate audit requirements, not assumed state-log fields. Define event thresholds, operating conditions, and a consistent tracking-controller interface before comparisons.

## 8. Example calculations

These are synthetic verification values, not vehicle specifications or dataset observations.

Take $B=1,K=2$, anchor $p^W_0=(10,20)$ m, $\psi_0=\pi/2$ rad, and future references $(10,22)$, $(9,24)$ m. Subtraction gives $(0,2)$, $(-1,4)$ m. At this angle, $R^\top$ maps $(a,b)$ to $(b,-a)$, so targets are $(2,0)$, $(4,1)$ m.

Use prediction $(2.5,-0.25)$ m for the first point, mask $(1,0)$, and synthetic scales $s=(2,0.5)$ m. The valid residual is $(0.5/2,-0.25/0.5)=(0.25,-0.5)$. Thus $L=(0.25^2+(-0.5)^2)/2=5/32=0.15625$. The second point contributes nothing, even if its stored target/prediction is nonfinite.

## 9. Expected outputs

The following synthetic cases were numerically verified before this research-only handoff was finalized. They provide expected values for independent implementation; no executable code is included.

| Check | Inputs or condition | Expected mathematical result |
|---|---|---|
| Frame transform | Section 8 anchor, heading, and future positions | Targets $(2,0),(4,1)$ metres; absolute tolerance $10^{-12}$ with float64 |
| Masked normalized loss | Section 8 prediction, mask, and scales | $L=5/32=0.15625$; absolute tolerance $10^{-12}$ with float64 |
| Prediction gradient | Same loss; first point valid, second invalid | $[(0.125,-1),(0,0)]$ in $\mathrm{m}^{-1}$ |
| Exact prediction | Prediction equals every valid target | Zero loss and zero prediction gradient |
| No supervision | All future labels invalid or no target points | Skip the objective and optimizer update; do not report zero error |
| Invalid active values | Nonfinite valid target or prediction | Reject the evaluation as undefined |
| Large-coordinate precision | Subtract $10^8$ m from $10^8+0.25$ m | Float64 preserves $0.25$ m; rounding both inputs to float32 first loses the offset |

The float64 evaluation of the Section 8 transformed-target fixture produced $0.1562500000000001$, within the stated loss tolerance. Direct evaluation of the stated local-coordinate fixture produced float32 loss $0.15625$ and the displayed prediction gradient exactly. A float64 finite-difference gradient check also passed. These numerical checks concern the stated equations; they are not measurements of a driving model or the Leaf.

## 10. Potential mathematical risks

Dead-reckoning drift and timing/calibration errors can dominate apparent model error. Rotation exists and is unique for a supplied valid heading modulo $2\pi$, but reference uncertainty is not removed by transforming coordinates. Very large global coordinates require sufficiently precise subtraction. A planar approximation may mislabel graded/banked roads.

For positive scales and fixed valid targets, the loss is convex in the predicted positions; it is not generally convex in neural parameters, and no training convergence guarantee is claimed. Tiny scales amplify errors and gradients. Unequal scales intentionally weight forward/lateral errors differently. Masking may preferentially remove difficult scenarios: report coverage by scenario, not just aggregate loss.

A mean trajectory can cut between feasible alternatives; low imitation loss does not guarantee appropriate braking or safe emergency handling. The diagnostic must not substitute for closed-loop evaluation or a verified tracker. Missing language supervision prevents any claim of learned instruction grounding, regardless of a transformer component's name.

## 11. Results in limiting / edge cases

- Eq. (1): zero heading gives ordinary subtraction; adding $2\pi$ changes nothing; a future position equal to the anchor maps to zero. These match rigid-frame geometry.
- As a future offset tends to zero for continuous motion, the displacement tends to zero. No division by the offset occurs here; zero/duplicate offsets remain outside the declared sampling contract.
- A stationary future maps to zero displacement. No velocity/curvature division occurs, so zero speed and straight motion introduce no singularity.
- Eq. (2): exact predictions give zero; one valid point averages its two normalized squared errors; repeated equal examples leave the mean unchanged.
- Missing future labels are excluded, including a partially observed horizon. All-missing labels or an empty verification input require SKIP; an empty model batch is not forwarded. Unknown masks are not treated as validity.
- Zero/nonfinite scales and nonfinite valid data are rejected. Invalid masked NaNs are never used in arithmetic. Heading wrap-around is handled by sine/cosine, without raw angle subtraction.

**Open questions for the user**

1. What is the dataset path, and which representative files describe synchronization, cameras, calibration, and state channels?
2. What motion reference exists: externally referenced poses, odometry, or only signals for dead reckoning? What validation/uncertainty information accompanies it?
3. Are paired language instructions/decisions available, or must they be collected/annotated for the VLA phase?
4. What highway speed, lighting/weather, road geometry, and following situations define the first operating domain? Which evaluation labels/simulator support lane and lead-relative metrics?
