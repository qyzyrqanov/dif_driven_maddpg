# Minimal Experimental Runs and Result Outputs for Revision

This document summarizes the experimental evidence and result artifacts needed for the targeted manuscript revision. The scope is limited to reviewer-requested issues: multiple seeds, same-environment baselines, environment-size generalization, computational cost, and consistency of reported quantitative evidence.

## 1. Experimental Run Set

### 1.1 Main multi-seed runs for the proposed method

These runs support the main reported results and replace the previous single-run evidence.

| Method | Team sizes | Environment | Seeds | Episodes per seed | Output window |
|---|---:|---:|---:|---:|---:|
| Full proposed method | n = 4, 5, 6 | 10 × 10 | 3 total | 1000 | Last 200 episodes |

If an existing original run has a known seed and usable logs, it can be treated as one of the three seeds. In that case, only two additional seeds are needed for each team size.

Expected long-run count:

| Case | New long training runs |
|---|---:|
| Original full-method logs usable as seed 0 | 6 |
| Original full-method logs not usable | 9 |

### 1.2 Same-environment learned baseline

This baseline addresses the request for within-environment comparison and strengthens the existing ablation evidence.

| Method | Team sizes | Environment | Seeds | Episodes per seed | Output window |
|---|---:|---:|---:|---:|---:|
| Shared actor–critic without assignment-aware reward shaping | n = 4, 5, 6 | 10 × 10 | 3 total | 1000 | Last 200 episodes |

This is the same ablation concept already present in the manuscript: removal of the progressive and distance-based Hungarian assignment reward terms while keeping the architecture and the remaining reward components unchanged.

Expected long-run count:

| Case | New long training runs |
|---|---:|
| Original ablation logs usable as seed 0 | 6 |
| Original ablation logs not usable | 9 |

### 1.3 Same-environment heuristic baseline

This is an evaluation-only baseline. It does not require RL training and is used only to provide a simple non-learning reference in the same environment.

| Method | Team sizes | Environment | Evaluation seeds | Episodes per seed | Training required? |
|---|---:|---:|---:|---:|---:|
| Hungarian assignment + proportional controller | n = 4, 5, 6 | 10 × 10 | 3 | 200 | No |

The heuristic baseline should be reported using task-level metrics only. Episode return is not necessary unless the same reward logger is applied consistently.

### 1.4 Environment-size generalization evaluation

This evaluation addresses the request for generalization to different environment sizes. The intended comparison is training in the original field and evaluating without retraining in a larger field.

| Policy | Training environment | Evaluation environment | Team sizes | Evaluation seeds | Episodes per seed | Training required? |
|---|---:|---:|---:|---:|---:|---:|
| Full proposed method | 10 × 10 | 10 × 10 | n = 4, 5, 6 | 3 | 200 | No additional training |
| Full proposed method | 10 × 10 | 15 × 15 | n = 4, 5, 6 | 3 | 200 | No additional training |

The 10 × 10 row is the in-distribution evaluation reference. The 15 × 15 row is the environment-size generalization test.

### 1.5 Computational cost logging

Computational cost does not require an additional scientific experiment, but the following quantities are needed from the training/evaluation process.

| Category | Needed quantity |
|---|---|
| Hardware | CPU model, GPU model if used, RAM, operating environment if relevant |
| Training cost | Mean training time per seed for n = 4, n = 5, n = 6 |
| Evaluation cost | Evaluation time for 200 episodes, if available |
| Inference cost | Mean inference time per agent per step |
| Model size | Actor parameter count, critic parameter count, total trainable parameters |
| Memory | Replay buffer size, peak RAM/GPU memory if available |
| Reproducibility | Number of seeds, episodes per seed, batch size, learning rates, hidden layers, activation functions |

## 2. Required Quantitative Result Tables

### 2.1 Main multi-seed summary table

This table replaces the single-run final summary table.

| n | Seeds | Success rate (%) | Completion time (steps) | Episode return | Collision-active steps (%) |
|---:|---:|---:|---:|---:|---:|
| 4 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 5 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 6 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |

Basis: last 200 episodes per seed.

### 2.2 Same-environment baseline comparison table

This table provides the direct comparison requested by the reviewers.

| n | Method | Seeds | Success rate (%) | Completion time (steps) | Collision-active steps (%) | Path length |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Full proposed method | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 4 | No-assignment-shaping baseline | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 4 | Hungarian + proportional controller | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 5 | Full proposed method | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 5 | No-assignment-shaping baseline | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 5 | Hungarian + proportional controller | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 6 | Full proposed method | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 6 | No-assignment-shaping baseline | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 6 | Hungarian + proportional controller | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |

Episode return can be excluded from this table because the heuristic controller is not a learned reward-optimizing method. If the same reward logger is applied to all methods, return can be added as an additional column.

### 2.3 Environment-size generalization table

This table addresses generalization to a larger field.

| n | Training field | Evaluation field | Seeds | Success rate (%) | Completion time (steps) | Collision-active steps (%) | Path length |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 10 × 10 | 10 × 10 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 4 | 10 × 10 | 15 × 15 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 5 | 10 × 10 | 10 × 10 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 5 | 10 × 10 | 15 × 15 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 6 | 10 × 10 | 10 × 10 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |
| 6 | 10 × 10 | 15 × 15 | 3 | mean ± SD | mean ± SD | mean ± SD | mean ± SD |

This table should be described as evaluation without retraining.

### 2.4 Computational cost and reproducibility table

This table addresses the reviewer request for computational cost analysis and missing reproducibility details.

| Item | Value |
|---|---:|
| CPU | value |
| GPU | value |
| RAM | value |
| Training episodes per seed | 1000 |
| Evaluation episodes | 200 |
| Number of seeds | 3 |
| Actor hidden layers | value |
| Critic hidden layers | value |
| Actor parameters | value |
| Critic parameters | value |
| Replay buffer size | 50,000 |
| Batch size | 128 |
| Training time per seed, n = 4 | mean ± SD |
| Training time per seed, n = 5 | mean ± SD |
| Training time per seed, n = 6 | mean ± SD |
| Inference time per agent per step | mean ± SD ms |
| Peak memory usage | value |

### 2.5 Updated statistical robustness table

If the manuscript keeps the current statistical robustness section, the statistical tables should be recomputed using the multi-seed data rather than single-run episode sequences.

Minimum updated statistical outputs:

| Output | Content |
|---|---|
| Success-rate uncertainty | 95% confidence intervals over seeds or bootstrap intervals over the final evaluation episodes |
| Completion-time uncertainty | mean ± SD and/or 95% confidence interval over seeds |
| Return uncertainty | mean ± SD and/or 95% confidence interval over seeds |
| Baseline comparison significance | statistical comparison between full method and no-assignment-shaping baseline, if included |

The existing Kruskal–Wallis, Mann–Whitney, Kaplan–Meier, log-rank, and block-bootstrap outputs can be retained only if they are recomputed consistently from the revised multi-seed dataset.

## 3. Required Plots

### 3.1 Updated learning-curve plot

This replaces the current single-run learning curve.

| Plot | Data | Curves | Uncertainty display |
|---|---|---|---|
| Learning curve | Episode return over training | n = 4, n = 5, n = 6 | Shaded variability across seeds |

Expected figure role: show training behavior of the full method across three seeds.

### 3.2 Updated rolling-success plot

This replaces the current single-run rolling success plot.

| Plot | Data | Curves | Uncertainty display |
|---|---|---|---|
| Rolling success rate | Rolling success over training, window = 50 | n = 4, n = 5, n = 6 | Shaded variability across seeds |

Expected figure role: show that final task success is not a single-seed artifact.

### 3.3 Optional baseline plot

A plot is not strictly necessary if the baseline comparison table is clear. If included, the minimal useful plot is a grouped bar chart.

| Plot | Data | Groups | Bars |
|---|---|---|---|
| Baseline comparison | Final-window success rate | n = 4, n = 5, n = 6 | Full method, no-assignment baseline, Hungarian heuristic |

This plot is optional because the table is the primary result artifact.

### 3.4 Optional generalization plot

A plot is not strictly necessary if the generalization table is clear. If included, the minimal useful plot is a grouped bar chart.

| Plot | Data | Groups | Bars |
|---|---|---|---|
| Environment-size generalization | Success rate or completion time | n = 4, n = 5, n = 6 | 10 × 10 evaluation, 15 × 15 evaluation |

This plot is optional because the table directly answers the reviewer request.

## 4. Existing Qualitative Figures

The current trajectory figures for n = 4, n = 5, and n = 6 can remain as qualitative illustrations. They do not need to be expanded for the minimal revision.

Needed adjustment:

| Existing qualitative output | Revision role |
|---|---|
| Representative trajectories for n = 4, 5, 6 | Retain as examples only; quantitative conclusions should refer to multi-seed tables |

No new qualitative trajectory figures are required for the baseline or 15 × 15 generalization evaluation unless the revised results need visual explanation.

## 5. Metric Set

The following metrics are sufficient for the revised quantitative results.

| Metric | Definition/use |
|---|---|
| Success rate | Fraction of episodes in which all agents complete coverage within the horizon |
| Completion time | First timestep at which all agents complete coverage; computed on successful episodes |
| Episode return | Scaled reward sum; used for learned policies and main training curves |
| Collision-active steps | Percentage of timesteps with nonzero collision penalty or collision event indicator |
| Path length | Total traveled distance across agents; useful for heuristic and baseline comparison |
| Training time | Wall-clock time per training run |
| Inference time | Policy forward-pass time per agent per step |
| Model parameters | Actor and critic parameter counts |
| Memory usage | Replay buffer size and peak memory usage if available |

## 6. Minimal Artifact List for Manuscript Update

The minimal revised result package consists of the following artifacts.

| Artifact | Type | Required? |
|---|---|---:|
| Multi-seed main summary table | Table | Yes |
| Baseline comparison table | Table | Yes |
| Environment-size generalization table | Table | Yes |
| Computational cost and reproducibility table | Table | Yes |
| Updated learning curves over seeds | Plot | Yes |
| Updated rolling success over seeds | Plot | Yes |
| Updated statistical robustness outputs | Table/plot | Yes, if the statistical section remains |
| Baseline success bar plot | Plot | Optional |
| Generalization bar plot | Plot | Optional |
| New qualitative trajectory figures | Plot | No |
| Obstacle-specific results | Table/plot | No, if obstacles are acknowledged as a limitation |
| n = 8 or n = 10 results | Table/plot | No, if scalability claims are narrowed and limitation is acknowledged |

## 7. Minimal New Long-Run Count Summary

| Scenario | Full method | No-assignment baseline | Total new long runs |
|---|---:|---:|---:|
| Existing original logs usable as one seed | 6 | 6 | 12 |
| Existing original logs not usable | 9 | 9 | 18 |

Evaluation-only runs for the heuristic baseline and 15 × 15 generalization do not count as long training runs.
