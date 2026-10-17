# LLM-A*: conditional computation-efficiency estimate

Date: 2026-10-03. These are analytical sensitivity estimates, not measured Viktor performance, statistical forecasts, or deadline guarantees. No implementation or training run was performed.

## 1. Problem definition

Estimate how much LLM-guided A* could reduce search work and complete camera-to-validated-plan latency on Viktor. Distinguish search-operation counts, search storage, neural inference, and the whole critical path.

Evidence: Table 1 of the supplied [LLM-A* paper](https://aclanthology.org/2024.findings-emnlp.60.pdf), transcribed and discussed in [the paper review](llm_astar_paper_review.md). Its strongest Llama 3 few-shot configuration reports operation ratio 0.4459, storage ratio 0.6402, relative path length 1.0247, and 100% validity on its benchmark. These are geometric-mean ratios across tasks, not timing measurements or guarantees for our adapted pseudocode.

## 2. Assumptions

1. The baseline and hybrid solve the same graph problems and meet comparable validity requirements. A faster failure or materially poorer path is not automatically a useful improvement.
2. Common camera, perception, mandatory structured reasoning, graph preparation, trajectory processing, and validation work is accounted for in both pipelines. Any change to that work must be charged explicitly.
3. For the numerical sensitivity examples only, search time is proportional to the reported operation count, giving a hypothetical search-time ratio of 0.4459. Different operation costs, queue behavior, collision checks, and our changed reopening/target rules can invalidate this assumption.
4. Example times and added overhead are assumed, not measured or fitted. The examples are neither confidence intervals nor p99 predictions.
5. Model guidance reuses the required fresh-scene reasoning computation only if the architecture actually shares that forward pass. A separate model invocation incurs its full extra latency and memory cost.
6. The timing model adds sequential critical-path durations. Concurrent implementations require measurement of their actual dependency path and contention; overlapping work must not be double-counted.

## 3. Notation

| Symbol | Meaning / role | Domain and units |
|---|---|---|
| $T_A$ | Ordinary A* search time; measured input when available | Positive scalar, ms |
| $T_C$ | Common non-search critical-path time, including common graph preparation; input | Nonnegative scalar, ms |
| $T_X$ | Extra guidance, filtering, queue-rekey, and other hybrid-only overhead; input | Nonnegative scalar, ms |
| $\rho$ | Guided/base search-time ratio excluding overhead in $T_X$; assumption | Nonnegative, dimensionless |
| $T_B,T_H$ | Baseline and hybrid total times; derived outputs | Positive scalars, ms |
| $\alpha,\beta$ | Baseline search fraction and extra-overhead fraction | $\alpha=T_A/T_B\in(0,1]$, $\beta=T_X/T_B\ge0$; dimensionless |
| $S,\Delta T$ | Total speedup factor and time saved | Dimensionless; ms |
| $\rho_W$ | Hypothetical dynamic weighted-A* time ratio, from operation ratio | 0.6091 in examples; dimensionless |
| $F,P_{\rm eff}$ | Core operation count and assumed effective compute rate | FLOP; FLOP/s |
| $N_w,b_w,BW$ | Active weight count, bytes per weight, memory bandwidth ceiling | Counts; byte/weight; byte/s |

The paper's operation ratio supplies the value used for $\rho$ in examples; it does not establish a true runtime ratio. No percentile is associated with any assumed time.

## 4. Derivation

The ordinary pipeline consists of common work plus search. The hybrid replaces that search with guided search and adds guidance-related overhead. Subtracting their totals identifies the break-even condition. Dividing by the baseline total gives an Amdahl-style speedup expression: only the fraction spent in search can benefit from reducing search work.

The stronger classical comparator is dynamic weighted A*. Under the same hypothetical count-to-time model, only the difference between its 0.6091 operation ratio and the hybrid's 0.4459 ratio is available to pay for learned guidance. This comparison is not quality-matched: weighted A* reports relative path length 100.24%, versus 102.47% for the best hybrid.

## 5. Final equations

$$
\begin{aligned}
T_B &= T_C+T_A,\\
T_H &= T_C+\rho T_A+T_X,\\
\Delta T &= (1-\rho)T_A-T_X.
\end{aligned}
\tag{1}
$$

$$
S=\frac{T_B}{T_H}
=\frac{1}{1-\alpha(1-\rho)+\beta}.
\tag{2}
$$

$$
\begin{aligned}
\text{Beat ordinary A*:}\quad&T_X<(1-\rho)T_A,\\
\text{Beat weighted A*:}\quad&T_X<(\rho_W-\rho)T_A.
\end{aligned}
\tag{3}
$$

With the example ratios, these allowances are $0.5541T_A$ and $0.1632T_A$, respectively. Equal times are break-even, not improvement. The idealized search-only factor is $1/0.4459=2.2426553$; it is not a full-pipeline speedup.

Conditional neural-resource floors are

$$
T_{\rm compute}\ge\frac{F}{P_{\rm eff}},\qquad
T_{\rm weight\ stream}\ge\frac{N_wb_w}{BW}.
\tag{4}
$$

The first relation requires a defensible effective compute ceiling, and the second assumes the specified weights are streamed from the memory supplying that bandwidth. Both exclude additional execution costs and return seconds; multiply by 1000 for milliseconds. They cannot be substituted for device benchmarks.

## 6. Constraints

The paper's 35.98% reduction in search storage does not imply a 35.98% reduction in total CPU/GPU memory. The learned model, features, graph representation, and temporary buffers must also be counted. Peak sparse GPU throughput does not predict a priority-queue search loop's latency.

Our proposed shared neural core has approximately 37.75 million parameters and 20.94 GFLOPs for one illustrative configuration. This excludes the vision encoder, embeddings, and other components. The paper's 8B model is not the assumed deployment architecture; all Viktor learned components still require training from scratch.

## 7. Evaluation procedure

1. Measure ordinary and weighted A* on representative graphs, including graph preparation and final validity checks.
2. Measure guided search with identical graph, collision, and timing rules. Charge guidance inference, filtering, reopening, and every queue rebuild.
3. Separately report full camera-to-published-plan latency, its tails, deadline misses, and valid-plan-by-deadline coverage over all admitted cycles.
4. Measure guidance generated by the shared model and by any proposed separate model as distinct configurations.
5. Compare path quality and feasibility at matched thresholds. The observed operation ratios need not match the paper's ratios.

This is an evaluation handoff, not an executed benchmark.

## 8. Example calculations

**Full-pipeline sensitivity.** Assume $T_C=20$ ms and $T_X=0.5$ ms. The 20 ms includes all common work; it is an independent illustrative scenario, not the existing stage-allocation worksheet. The 0.5 ms is a proposed sensitivity value, not an established cost for a guidance head.

| Ordinary search $T_A$ | Baseline total $T_B$ | Hypothetical guided total $T_H$ | Latency reduction | Total speedup |
|---:|---:|---:|---:|---:|
| 1 ms | 21 ms | 20.9459 ms | 0.2576% | 1.0026x |
| 5 ms | 25 ms | 22.7295 ms | 9.0820% | 1.0999x |
| 10 ms | 30 ms | 24.9590 ms | 16.8033% | 1.2020x |

**Extra-inference sensitivity.** Keeping $T_C=20$ ms and $T_A=5$ ms, increase added overhead:

| Extra overhead $T_X$ | Guided total | Latency reduction relative to 25 ms |
|---:|---:|---:|
| 0.5 ms | 22.7295 ms | 9.082% |
| 2 ms | 24.2295 ms | 3.082% |
| 6 ms | 28.2295 ms | -12.918%; slower |

The break-even overhead is 2.7705 ms against ordinary A*. Against weighted A*, it is only 0.816 ms. With 0.5 ms overhead, weighted A* takes a hypothetical 23.0455 ms and the hybrid 22.7295 ms: a saving of 0.3160 ms, or about 1.3712%, with unmatched path quality.

**Relationship to our 20-30 ms worksheet.** The [existing latency note](low_latency_architecture.md) allocates 25 ms to named stages and 5 ms to queue/contention/unassigned allowance for the 30 ms target. It contains no explicit graph-building or guided-search allocation. Adding a hypothetical 2.2295 ms guided search and 0.5 ms hybrid overhead consumes 2.7295 ms of that allowance, leaving 2.2705 ms for previously unbudgeted graph work, contention, and margin. It cannot be appended to an already spent 30 ms budget.

The 20 ms worksheet has only 3 ms unassigned. The same insertion leaves 0.2705 ms before unbudgeted graph work and contention. Neither worksheet is a latency forecast; camera, model, trajectory-processing, and validation allowances remain unmeasured, and the geometric-search adaptation may change them.

**Neural computation scale.** For the illustrative 20.937965568 GFLOP core, assumed effective rates of 5, 10, and 20 TFLOP/s give compute floors of 4.187593, 2.093797, and 1.046898 ms. These are arithmetic screens, not Thor performance measurements.

For contrast, one full FP16 stream of 8 billion weights is 16 GB. At Thor's published 273 GB/s bandwidth, that stream alone takes at least 58.608059 ms under the streaming assumption, before other work. This is why deploying the paper's large 16-bit model unchanged would conflict with the current budget under that memory-traffic model. [NVIDIA Thor specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)

## 9. Expected outputs

All arithmetic in Section 8 was checked with an ephemeral Python calculation. No source files or search implementation were created. The tabulated six-decimal millisecond values have rounding tolerance $10^{-6}$ ms; displayed percentages and factors use their stated rounding precision. Numerical precision does not represent predictive confidence.

Key checks: $20+0.4459(5)+0.5=22.7295$ ms; $(1-0.4459)(5)=2.7705$ ms; $(0.6091-0.4459)(5)=0.816$ ms; and $1000(8\times10^9)(2)/(273\times10^9)=58.608058608\ldots$ ms.

## 10. Potential mathematical risks

Geometric-mean operation ratios cannot predict individual cases, arithmetic-mean total runtime, or tail latency. A conditional quality bound on returned paths does not bound node expansions or guarantee finding a plan before the deadline. Learned guidance may reduce search work on large obstacle-rich maps while adding overhead on simple highway scenes.

The current parameters, camera configuration, graph size, collision-check cost, and learned-guidance quality are unmeasured. The paper's pretrained-model results do not forecast the quality of a small randomly initialized model after training. Reopening and global reprioritization can change the operation ratio from the published value. Energy efficiency requires power measurements; latency ratios alone do not supply joules per plan.

## 11. Results in limiting / edge cases

- If search is a negligible fraction of the pipeline, speeding it up has negligible benefit; any added guidance overhead may make the total slower.
- If $T_X=0$ and $T_C=0$, the entire runtime would improve by $1/\rho$. These conditions do not describe Viktor's full pipeline.
- If $\rho\ge1$ and $T_X\ge0$, guidance cannot improve time under this additive model.
- If $T_X=(1-\rho)T_A$, the hybrid merely breaks even with ordinary A*.
- If $\rho_W\le\rho$, nonnegative added overhead cannot beat weighted A* under the ratio model.
- Meeting a deadline with invalid reasoning or an invalid path is not successful completion.

## Open questions for the user

No answer is required for these conditional estimates. A hardware-specific forecast requires measured shared-model latency, ordinary/weighted search time on the selected graph, incremental guidance cost, camera timing, and the accepted quality/deadline criterion.
