# LLM-A* versus Alpamayo 2 Super: mathematical principles

Reviewed: 2026-10-03. Research only; no implementation, model execution, or latency measurement.

**Scope:** /home/barry/alpamayo2 at HEAD beb2977d9a7e9d66837d4a3ad5144ff59de37519. Its [README](/home/barry/alpamayo2/README.md:12) describes **Alpamayo 2 Super**, a 32B vision-language model plus a 2B action expert. These sizes are repository claims, not parameter counts independently measured from weights. The pre-existing SECURITY.md modification was left untouched.

**Paper:** Meng et al., [LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning](https://aclanthology.org/2024.findings-emnlp.60/), Findings of EMNLP 2024; local source /home/barry/Downloads/2024.findings-emnlp.60.pdf. The earlier [paper review](llm_astar_paper_review.md) contains the benchmark discussion. This comparison concerns the published algorithm, not every adaptation in our [Viktor pseudocode](llm_astar_pseudocode.md).

**Central finding:** LLM-A* uses generated waypoint coordinates to change the order of an explicit graph search. Alpamayo generates reasoning from visual/history/text context, then transports Gaussian noise into a continuous acceleration-curvature sequence using a learned vector field. A numerical vehicle model converts those controls into a trajectory.

## 1. Problem definition

| Aspect | LLM-A* paper | Alpamayo 2 Super local implementation |
|---|---|---|
| Input | Known static obstacle geometry, start and goal, expressed to an LLM | Camera history, ego pose history, prompt; navigation in the advanced demo |
| Neural contribution | Ordered waypoint suggestions | Reasoning context and a field over action sequences |
| Main computation | Priority-queue search on discrete states | Autoregressive text generation followed by whole-horizon neural flow updates |
| Planning representation | Geometric path through search states | Continuous acceleration and curvature at fixed future times |
| Inference objective | Search costs and a waypoint-biased priority | Integration of a learned conditional flow |
| Environmental validity | Depends on explicit graph obstacle/transition checks | No explicit collision checker in the inspected generation path |
| Vehicle model | Not supplied by the paper | Unicycle kinematic rollout |
| Optimality | Paper does not guarantee a shortest path | No shortest-path or constrained-control optimum provided by the sampler |
| Language role | Guides where search spends work | Conditions the trajectory expert through cached transformer features |

The methods address different levels of driving. Neither inspected method supplies a complete Nissan Leaf trajectory-following controller.

Alpamayo's validated trajectory profile selects six cameras and four frames each. The loader supplies 16 historical poses at 0.1 s intervals, ending at the current ego frame. The standard helper passes images and ego history; loaded calibration is not an explicit additional tensor in its model-input dictionary. This does not establish support for arbitrary fisheye arrangements. Sources: [input profile](/home/barry/alpamayo2/src/alpamayo2_super/input_profiles.py:40), [history/frame preparation](/home/barry/alpamayo2/src/alpamayo2_super/load_physical_aiavdataset.py:118), [input helper](/home/barry/alpamayo2/src/alpamayo2_super/helper.py:98).

## 2. Assumptions

1. Graph edge costs, goal heuristic, and waypoint attraction have compatible units. The paper's map units are not established as meters.
2. Graph validity refers to supplied geometry and transition rules; it does not automatically include vehicle footprint, moving traffic, or uncertain perception.
3. Alpamayo inputs use the current ego frame. Its rollout starts planar position and heading at zero and estimates initial speed from history.
4. Action normalization scales are positive. Constructor defaults are not verified settings of a loaded checkpoint: configuration and stored buffers can override them.
5. Physical time and flow time are distinct. Changing flow-step count changes generation computation, not the physical prediction horizon.
6. The probability factorization below describes the architecture; the API does not evaluate its trajectory density.
7. Source losses establish available training primitives, not the full training history or a reproducible training-from-scratch recipe.
8. Timing equations describe dependencies, not measured Thor performance. Viktor's deadline includes fresh cameras, structured reasoning, planning, validation, and publication.

## 3. Notation

| Symbol | Meaning / role | Domain or shape | Units |
|---|---|---|---|
| $\mathcal G=(V,E)$ | Search graph; input | Finite states and allowed edges | Counts dimensionless |
| $s,s_0,q,w$ | State, start, goal, active waypoint | Paper examples: 2D coordinates | Map length units |
| $c(u,v),g(s),h(s),f(s)$ | Edge cost, discovered cost, goal heuristic, priority | Scalars | Compatible cost units |
| $P,J(P)$ | Geometric path and summed edge cost | State sequence; scalar | Path cost units |
| $C$ | Camera/history/prompt conditioning | Mixed tensors and tokens | Images/tokens and physical pose inputs |
| $r=(r_1,\ldots,r_L)$ | Generated reasoning/context tokens | Discrete sequence | Token IDs dimensionless |
| $K(C,r)$ | Transformer key/value context cache | Per-layer tensors | Learned feature units |
| $\theta,\phi$ | VLM and expert parameters | Parameter collections | Model dependent |
| $N,\Delta t$ | Future intervals and physical interval duration | Class defaults $64$ and $0.1$ | Count; $\mathrm{s}$ |
| $a_k,\kappa_k$ | Longitudinal acceleration and curvature | $k=0,\ldots,N-1$ | $\mathrm{m}\,\mathrm{s}^{-2}$; $\mathrm{m}^{-1}$ |
| $\mu_a,\sigma_a,\mu_\kappa,\sigma_\kappa$ | Normalization means/scales | Scalars; scales positive | Corresponding action units |
| $U$ | Normalized control sequence | $\mathbb R^{N\times2}$ | Dimensionless |
| $\epsilon,Z_\tau$ | Gaussian noise and intermediate action tensor | $\mathbb R^{N\times2}$ | Dimensionless |
| $\tau,\Delta\tau$ | Flow time and solver step | $[0,1]$; positive scalar | Dimensionless |
| $v_\phi$ | Learned velocity in action space | $\mathbb R^{N\times2}$ | Normalized action per unit flow time |
| $M$ | Flow steps | Positive integer; class default 10 | Dimensionless |
| $x_k,y_k,\psi_k,v_k$ | Position, heading, speed at $k\Delta t$ | Scalars | $\mathrm{m}$; $\mathrm{rad}$; $\mathrm{m}\,\mathrm{s}^{-1}$ |
| $X=F(U,v_0)$ | Decoded pose trajectory | Positions $N\times3$, rotations $N\times3\times3$ | $\mathrm{m}$; rotation matrices dimensionless |
| $B,S,J$ | Batch size, trajectory sets, samples per set | Positive integers | Dimensionless |
| $\mathcal I_f,\mathcal I_o$ | Selected future-token and other-token label indices | Finite sets | Dimensionless |
| $\lambda_f,\lambda_o$ | Token-loss weights | Nonnegative scalars | Dimensionless |
| $\gamma$ | Navigation flow-guidance weight | Scalar | Dimensionless |

The neural field $v_\phi$ and physical car speed $v_k$ are different quantities. $J(P)$ denotes path cost, whereas the unadorned tensor dimension $J$ denotes sample count.

## 4. Derivation

### 4.1 LLM-A*: changing search order

The reference graph problem is:

$$
\begin{aligned}
P^\star&\in\operatorname*{arg\,min}_{P:s_0\leadsto q}J(P),\\
J(P)&=\sum_{(u,v)\in P}c(u,v).
\end{aligned}
$$

Transitions must belong to the valid graph. This defines the optimum against which a returned path can be compared, not a claim that LLM-A* attains it.

Ordinary A* selects the frontier state minimizing $g(s)+h(s)$. The paper adds attraction to the current LLM waypoint:

$$
f_{\mathrm{LLM\text{-}A^\star}}(s)
=g(s)+h(s)+d(s,w).
\tag{1}
$$

For Euclidean distances, $h(s)=\lVert s-q\rVert_2$ and $d(s,w)=\lVert s-w\rVert_2$. Edge relaxation still uses:

$$
g_{\mathrm{candidate}}(v)=g(u)+c(u,v).
\tag{2}
$$

Thus waypoint distance changes priority, not accumulated edge cost. It is not a hard requirement that the reconstructed path visit every waypoint.

The paper inserts missing start/goal points, removes suggestions inside obstacles, advances its target during search, and recomputes OPEN priorities when targets change. Algorithm 1 changes target when encountering a matching neighbor, before the subsequent CLOSED/obstacle checks. Our earlier pseudocode uses different, more conservative update rules.

The augmented heuristic can overestimate remaining goal cost. If the target becomes the goal and both distances coincide:

$$
f(s)=g(s)+2h(s).
\tag{3}
$$

The paper acknowledges loss of shortest-path optimality. A conditional suboptimality analysis needs further metric and search-mechanics assumptions; the earlier paper review discusses such a derivation without attributing it to the paper.

Source: paper Sections 3.1–3.2, Algorithm 1, Appendix A.

### 4.2 Alpamayo: conditional reasoning and action generation

A useful architectural factorization is:

$$
p_{\theta,\phi}(r,U\mid C)
=p_\theta(r\mid C)\,
p_\phi\!\left(U\mid K(C,r)\right).
\tag{4}
$$

The VLM generates text autoregressively. Its cache contains visual/history/prompt features and generated reasoning context. The expert therefore receives more than a textual explanation alone.

The standard wrapper generates up to the future-trajectory boundary, then invokes the expert. It masks discrete trajectory vocabulary during text generation. Continuous future controls come from the expert rather than autoregressively generated coordinate tokens.

Sources: [generation and boundary](/home/barry/alpamayo2/src/alpamayo2_super/models/alpamayo2_super.py:281), [cache-conditioned expert](/home/barry/alpamayo2/src/alpamayo2_super/models/alpamayo2_super.py:325).

Equation (4) does not state that the trajectory minimizes an explicit driving cost. Conditioning on generated reasons also does not establish their truth or causal faithfulness.

### 4.3 Flow-matching supervision

Ground-truth poses are converted into controls and normalized:

$$
U_{k,1}=\frac{a_k-\mu_a}{\sigma_a},
\qquad
U_{k,2}=\frac{\kappa_k-\mu_\kappa}{\sigma_\kappa}.
\tag{5}
$$

Training samples Gaussian noise $\epsilon\sim\mathcal N(0,I)$ and flow time $\tau$, then constructs:

$$
Z_\tau=(1-\tau)\epsilon+\tau U,
\qquad
\frac{\partial Z_\tau}{\partial\tau}=U-\epsilon.
\tag{6}
$$

The exact elementwise-mean MSE can be written:

$$
\mathcal L_{\mathrm{FM}}
=\mathbb E\left[
\frac{1}{2N}
\left\lVert
v_\phi\!\left(Z_\tau,\tau;K(C,r)\right)
-(U-\epsilon)
\right\rVert_F^2
\right].
\tag{7}
$$

The expectation includes training examples, noise, and flow time. The conditioning-label recipe is not established by the checkout's incomplete training pipeline.

An ideal unlimited-capacity squared-error predictor is the conditional expectation:

$$
v^\star(z,\tau;K)
=\mathbb E[U-\epsilon\mid Z_\tau=z,\tau,K].
\tag{8}
$$

Equation (8) is our derivation of the loss minimizer. Straight training bridges do not imply that each inference sample follows a known straight line to a preselected true future. The learned field represents a conditional distribution.

The class defaults to $\tau=0.999(1-b)$ with $b\sim\operatorname{Beta}(1.5,1)$; uniform sampling is also supported. Its stored guidance-dropout rate is not used by this constructor, which returns no dropout mask.

Source: [target construction and MSE](/home/barry/alpamayo2/src/alpamayo2_super/diffusion/flow_matching.py:151).

### 4.4 Whole-horizon flow inference

Inference approximates the ODE:

$$
\frac{dZ_\tau}{d\tau}=v_\phi(Z_\tau,\tau;K),
\qquad Z_0\sim\mathcal N(0,I).
\tag{9}
$$

A sampler-specific temperature can rescale the initial Gaussian; its default is one. It is distinct from the wrapper's text-sampling temperature.

The implemented Euler updates are:

$$
\begin{aligned}
\tau_j&=\frac{j}{M},\qquad \Delta\tau=\frac{1}{M},\\
Z_{j+1}&=Z_j+\Delta\tau\,v_\phi(Z_j,\tau_j;K),\\
\widehat U&=Z_M.
\end{aligned}
\tag{10}
$$

Each expert pass updates all $N$ future control pairs. There are $M$ sequential passes over a whole horizon, not $N$ autoregressive control-generation steps. The context cache is reused, and new expert-token cache entries are cropped after each pass. This offers parallelism across future controls, but does not remove the sequential text-generation and flow dependencies.

Sources: [Euler implementation](/home/barry/alpamayo2/src/alpamayo2_super/diffusion/flow_matching.py:118), [expert pass/cache cropping](/home/barry/alpamayo2/src/alpamayo2_super/models/alpamayo2_super.py:353).

The flow loop does not evaluate $-\nabla_U J_{\mathrm{collision}}$ or minimize an explicit obstacle-avoidance objective. A learned generative field is not automatically a driving-cost gradient.

### 4.5 Actual vehicle integration

After denormalization, the conceptual continuous model is:

$$
\begin{aligned}
\dot x&=v\cos\psi,&\qquad \dot y&=v\sin\psi,\\
\dot v&=a,&\qquad \dot\psi&=v\kappa.
\end{aligned}
\tag{11}
$$

The discrete implementation is:

$$
\begin{aligned}
v_{k+1}&=v_k+a_k\Delta t,\\
\psi_{k+1}
&=\psi_k+\kappa_k
\left(v_k\Delta t+\tfrac{1}{2}a_k\Delta t^2\right),\\
x_{k+1}
&=x_k+\tfrac{\Delta t}{2}
\left(v_k\cos\psi_k+v_{k+1}\cos\psi_{k+1}\right),\\
y_{k+1}
&=y_k+\tfrac{\Delta t}{2}
\left(v_k\sin\psi_k+v_{k+1}\sin\psi_{k+1}\right).
\end{aligned}
\tag{12}
$$

It starts at zero planar position/yaw in the ego frame. Initial speed is estimated from history unless supplied through the lower-level action-space interface. Output height remains the last historical height. This is a planar kinematic model embedded in 3D pose tensors.

Source: [action-to-trajectory rollout](/home/barry/alpamayo2/src/alpamayo2_super/action_space/unicycle_accel_curvature.py:326).

Defaults $N=64$ and $\Delta t=0.1\,\mathrm{s}$ produce poses at $(0.1,\ldots,6.4)\,\mathrm{s}$. In contrast, $M=10$ produces a flow-time grid from zero to one in increments of $0.1$. The repeated numerical value has different units and roles.

### 4.6 Separate training primitives

The wrapper's training-style forward computes next-token losses, separately averaged over future-trajectory and other selected labels:

$$
\begin{aligned}
\operatorname{CE}(\mathcal I)
&=-\frac{1}{\lvert\mathcal I\rvert}
\sum_{i\in\mathcal I}\log p_\theta(y_i\mid C,y_{<i}),\\
\mathcal L_{\mathrm{VLM}}
&=\lambda_f\operatorname{CE}(\mathcal I_f)
+\lambda_o\operatorname{CE}(\mathcal I_o).
\end{aligned}
\tag{13}
$$

Here $y_i$ is a supervised token. The helper returns zero for an empty selected set.

The expert has a separate forward that converts future poses into controls and computes Equation (7) using externally supplied VLM outputs. The wrapper's forward does not call it. A combined loss such as $\mathcal L_{\mathrm{VLM}}+\eta\mathcal L_{\mathrm{FM}}$, with a chosen weight $\eta$, is therefore a possible training design, not a combined objective established by the inspected wrapper.

Sources: [wrapper loss](/home/barry/alpamayo2/src/alpamayo2_super/models/alpamayo2_super.py:197), [selected-label CE](/home/barry/alpamayo2/src/alpamayo2_super/models/utils.py:180), [expert loss](/home/barry/alpamayo2/src/alpamayo2_super/models/expert.py:123). No complete optimizer/trainer entrypoint was found.

The delta tokenizer rounds and clips coordinate increments into bins. These trajectory tokens are not A* graph nodes or a search lattice. Source: [delta tokenizer](/home/barry/alpamayo2/src/alpamayo2_super/models/delta_tokenizer.py:87).

### 4.7 Two meanings of “guidance”

LLM-A* guidance is the scalar distance bias in Equation (1). Alpamayo's advanced navigation example combines neural fields:

$$
v_{\mathrm{CFG}}
=(1-\gamma)v_{\mathrm{no\ nav}}+\gamma v_{\mathrm{nav}}.
\tag{14}
$$

Both branches retain camera/history information. The no-navigation branch also replays the guided branch's generated reasoning. It is not fully unconditional and retains navigation information mediated through that reasoning.

At $\gamma=1$ the result is the guided field; larger weights extrapolate away from the other field. This is not a hard navigation or collision constraint.

The ordinary expert constructor rejects configurations enabling CFG. The separate two-GPU demo supplies both callbacks and overrides it at sampling time. Both fields are evaluated even at $\gamma=1$.

Sources: [field combination](/home/barry/alpamayo2/src/alpamayo2_super/diffusion/flow_matching.py:94), [constructor restriction](/home/barry/alpamayo2/src/alpamayo2_super/models/expert.py:84), [retained guided reasoning](/home/barry/alpamayo2/examples/two_gpu_nav_cfg_demo.py:406), [runtime override](/home/barry/alpamayo2/examples/two_gpu_nav_cfg_demo.py:502).

### 4.8 Pose-to-action inversion is regularized estimation

Action labels are estimated from poses using smoothing and least squares. A representative solver minimizes:

$$
\begin{gathered}
\lVert W^{1/2}(Az-b)\rVert_2^2
+z^\top Rz+\rho\lVert z\rVert_2^2,\\
(A^\top WA+R+\rho I)z=A^\top Wb.
\end{gathered}
\tag{15}
$$

Here $z\in\mathbb R^N$ is the unknown sequence, $A=\operatorname{diag}(s)$ contains interval coefficients, $b\in\mathbb R^N$ contains observations, $W$ contains data weights, $R$ is a finite-difference smoothing matrix, and $\rho$ is ridge regularization. Units depend on the fitted quantity and must balance the objective.

For acceleration, $A=I$ and $b_k=\frac{v_{k+1}-v_k}{\Delta t}$. For curvature, $s_k=v_k\Delta t+\tfrac{1}{2}a_k\Delta t^2$ and $b_k=\psi_{k+1}-\psi_k$, so $s_k\kappa_k$ is an angular increment.

The control fits use second-order smoothing. Penalizing second differences of acceleration is not the same as penalizing jerk itself. The numerical solver uses Cholesky, not an explicit matrix inverse. Curvature labels are clipped and set to zero around low-speed intervals in the pose-to-action conversion.

Sources: [regularized solver](/home/barry/alpamayo2/src/alpamayo2_super/action_space/utils.py:241), [acceleration fitting](/home/barry/alpamayo2/src/alpamayo2_super/action_space/unicycle_accel_curvature.py:127), [curvature fitting/clipping](/home/barry/alpamayo2/src/alpamayo2_super/action_space/unicycle_accel_curvature.py:166).

Consequently pose→controls→pose is not generally an exact round trip. These fitting routines prepare targets and estimate speed; they are not online collision-avoidance optimization.

## 5. Final equations and distinction

$$
\begin{aligned}
\text{LLM-A*:}\quad
&(\mathcal G,s_0,q)
\longrightarrow\text{LLM waypoints}\\
&\longrightarrow\min_{\mathrm{OPEN}}\{g+h+d_w\}
\longrightarrow P,\\[1ex]
\text{Alpamayo:}\quad
&C\longrightarrow r,K(C,r)\\
&\longrightarrow\left[
\begin{gathered}
Z_0\sim\mathcal N(0,I),\\
Z_{j+1}=Z_j+\Delta\tau v_\phi(Z_j,\tau_j;K)
\end{gathered}
\right]\\
&\longrightarrow\widehat U
\longrightarrow X=F(\widehat U,v_0).
\end{aligned}
\tag{16}
$$

LLM-A* changes which discrete candidate is explored next. Alpamayo changes the numerical values of an entire candidate control sequence at each flow step.

Neither system's generated language is itself a validity certificate. LLM-A* relies on explicit search over supplied geometry. Alpamayo's learned conditioning shapes an action distribution, and its decoder imposes a particular kinematic relationship.

## 6. Constraints and guarantees

The action-space class declares default acceleration bounds $[-9.8,9.8]\,\mathrm{m}\,\mathrm{s}^{-2}$ and curvature bounds $[-0.33,0.33]\,\mathrm{m}^{-1}$. It provides a boolean bounds checker. The generation path directly decodes sampled controls without calling that check, and the rollout does not clamp them. No callers of the checker were found.

Sources: [bounds/checker](/home/barry/alpamayo2/src/alpamayo2_super/action_space/unicycle_accel_curvature.py:41), [sample-to-rollout path](/home/barry/alpamayo2/src/alpamayo2_super/models/alpamayo2_super.py:371).

Even enforced independent bounds would not establish vehicle feasibility. The unicycle approximation gives:

$$
\begin{aligned}
a_{\mathrm{lat}}&=v^2\kappa,\\
\lvert\kappa\rvert&\le\frac{a_{\mathrm{lat,max}}}{v^2}
\quad(v\ne0).
\end{aligned}
\tag{17}
$$

Here $a_{\mathrm{lat,max}}$ is a chosen admissible lateral acceleration, in $\mathrm{m}\,\mathrm{s}^{-2}$; this analysis does not establish its Nissan Leaf value. At $30\,\mathrm{m}\,\mathrm{s}^{-1}$, curvature $0.33\,\mathrm{m}^{-1}$ corresponds to $297\,\mathrm{m}\,\mathrm{s}^{-2}$ in this model. This illustrates why independent bounds are insufficient; it is not a claim that the model generates that combination.

A driving feasible set must account for swept-footprint collisions, lane/traffic constraints, headway, speed-coupled limits, actuation, and uncertainty. The inspected loss and rollout do not supply those constraints. The unicycle model also omits tire forces, slip, and detailed actuation.

LLM-A* could reject transitions using such a feasible set if an appropriate graph were constructed. The paper's static position graph does not already contain those vehicle constraints. Its benchmark validity cannot be transferred directly to camera-based highway driving.

## 7. Algorithm descriptions and computation

**LLM-A* sequence:** describe the map; generate/clean waypoint suggestions; initialize OPEN/CLOSED; select minimum-priority states, inspect valid neighbors, relax costs, and update target-dependent priorities; reconstruct a path.

**Alpamayo sequence:** prepare images/history/prompt; prefill the VLM; generate reasoning tokens; keep the context cache; sample a full action tensor; run $M$ flow passes; denormalize; estimate initial speed and integrate; return trajectories and text.

For standard Alpamayo inference, the dependency-based timing model is:

$$
\begin{aligned}
T_{\mathrm{Alpa}}
&\approx T_{\mathrm{camera/preprocess}}
+T_{\mathrm{vision/prefill}}\\
&\quad+\sum_{\ell=1}^{L}T_{\mathrm{text},\ell}
+\sum_{j=1}^{M}T_{\mathrm{expert},j}\\
&\quad+T_{\mathrm{rollout/output}}.
\end{aligned}
\tag{18}
$$

Terms denote successive critical-path stages, measured without double-counting any overlap. The CFG demo adds two expert evaluations per flow step and extra context construction/transfers. Sample count and context length also affect cost; shared prefill does not remove those costs.

For LLM-A*:

$$
\begin{aligned}
T_{\mathrm{LLM\text{-}A^\star}}
&=T_{\mathrm{map/input}}
+T_{\mathrm{waypoint\ generation}}\\
&\quad+T_{\mathrm{search}}
+T_{\mathrm{target\ updates/output}}.
\end{aligned}
\tag{19}
$$

The paper supplies the map. A camera implementation must additionally construct its planning representation, predict moving obstacles, assign physical time, and validate the result.

The paper's strongest search-operation ratio is 44.59% of ordinary A*. This cannot be interpreted as a 55.41% reduction in Alpamayo inference, whose computation is different. The [efficiency note](llm_astar_efficiency_estimate.md) gives the break-even condition including learned-guidance overhead.

The README reports roughly 67 GiB and 71 GiB on two H100s for one six-camera/four-frame CFG configuration. This is a documented memory observation, not a latency measurement performed here or evidence of 20–30 ms operation on Thor. Source: [demo report](/home/barry/alpamayo2/README.md:201).

## 8. Example calculations

Checked using standalone Python arithmetic; no model code was executed.

**Changing search order.** Let goal $q=(10,0)$ and guide $w=(0,5)$. Suppose OPEN contains $A=(4,0)$ with $g(A)=4$ and $B=(2,3)$ with $g(B)=5$.

| Candidate | $g+h$ | Guide distance | $g+h+d_w$ |
|---|---:|---:|---:|
| $A$ | 10.000000 | 6.403124 | 16.403124 |
| $B$ | 13.544004 | 2.828427 | 16.372431 |

Ordinary A* selects $A$; waypoint-biased search selects $B$. Candidate coordinates do not change.

**Changing controls.** For one normalized pair, $U=(0.8,-0.2)$, $\epsilon=(-0.4,0.6)$, and $\tau=0.25$ give $Z_\tau=(-0.1,0.4)$ and target velocity $(1.2,-0.8)$. A step $\Delta\tau=0.1$ using that target gives $(0.02,0.32)$. This is an illustrative known training bridge; inference does not know its future endpoint $U$.

**Physical rollout.** For $v_0=10\,\mathrm{m}\,\mathrm{s}^{-1}$, $a_0=0$, $\kappa_0=0.1\,\mathrm{m}^{-1}$, and $\Delta t=0.1\,\mathrm{s}$:

$$
\begin{aligned}
\psi_1&=0.1\,\mathrm{rad},\\
(x_1,y_1)&=(0.99750208,\ 0.04991671)\,\mathrm{m}.
\end{aligned}
$$

The exact constant-curvature arc gives $(0.99833417,0.04995835)\,\mathrm{m}$. The Cartesian integrator is an approximation even for constant controls.

## 9. Expected outputs and evaluation

Alpamayo returns positions $[B,S,J,N,3]$, rotations $[B,S,J,N,3,3]$, and optional text arrays $[B,S,J]$. Its returned log-probability tensor is all zeros: a placeholder, not evaluated likelihood or calibrated confidence for selecting trajectories.

Source: [output assembly](/home/barry/alpamayo2/src/alpamayo2_super/models/alpamayo2_super.py:396).

The smoke example computes:

$$
\operatorname{minADE}
=\min_j\frac{1}{N}\sum_{k=1}^{N}
\left\lVert
\widehat p^{(j)}_{k,xy}
-p^{\mathrm{reference}}_{k,xy}
\right\rVert_2.
\tag{20}
$$

Here $j$ indexes samples and the reference is recorded future position. With one sample, it reduces to ADE. The reference is available offline; this metric is not an online selector, collision test, or closed-loop control evaluation. Source: [smoke metric](/home/barry/alpamayo2/src/alpamayo2_super/inference_smoke.py:139).

LLM-A* instead evaluates geometric length, search operations/storage, and map-path validity. Those metrics cannot be directly compared with imitation error.

## 10. Risks and implications for Viktor

**My interpretation:** borrow LLM-A*'s separation between learned suggestions and explicit feasibility, and Alpamayo's continuous whole-horizon control representation. Neither the 34B text-plus-flow implementation nor the static 2D graph is established to meet Viktor's requirements unchanged.

A possible highway-task formulation is:

$$
\widehat U\in\operatorname*{arg\,min}_{U\in\mathcal F(C)}
\left[
\begin{aligned}
&J_{\mathrm{lane}}(U;C)
+J_{\mathrm{following}}(U;C)\\
&\quad+J_{\mathrm{comfort}}(U)
+J_{\mathrm{learned}}(U;C)
\end{aligned}
\right].
\tag{21}
$$

Here $\mathcal F(C)$ is an explicitly constructed feasible set. The terms score lane tracking, following, comfort, and learned preferences, with compatible scaling. This is a proposed research direction, not an objective implemented by either source. Searching a finite candidate set finds only its best evaluated member, not necessarily a continuous global optimum.

For 20–30 ms, consequential choices are encoder size, serial reasoning-token count, flow/search iterations, candidate count, and validation cost. Fixed-shape structured reasoning and a small trajectory model are hypotheses to benchmark, not evidence of matching Alpamayo's capabilities. Required evidence, decisions, and reasons must remain in the same deadline.

The inspected API also does not explicitly output other vehicles' predicted trajectories. Their behavior can influence latent features, but Viktor's explicit trajectory-prediction requirement needs its own output contract and evaluation.

## 11. Limits

- Read-only inspection and independent arithmetic checks; no checkpoint loading, training, GPU profiling, or driving experiments.
- Constructor and loader defaults were distinguished from actual checkpoint configuration.
- Available losses do not establish the complete released training procedure.
- No collision-free, shortest-path, physical-feasibility, calibrated-reasoning, or 20–30 ms guarantee inferred from generated trajectories.
- Findings apply to this Alpamayo 2 Super checkout and this LLM-A* paper.

## Open questions

1. What action normalization, expert architecture, and attention settings are in the checkpoint used for comparison?
2. What synchronization, resolution, frame rate, and fisheye calibration will Viktor use?
3. Which structured evidence/reason labels accompany the recorded poses and vehicle states?
4. What validated vehicle limits and following-distance constraints define $\mathcal F(C)$?
5. What are complete-pipeline tail latencies on Thor under sustained load?
6. Can a compact model trained from scratch improve on a simple lane/following planner at matched validity and total computation?
