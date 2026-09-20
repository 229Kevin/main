# S0-S7 Spatial Mamba Ablation Design

## Goal

Implement eight reproducible spatial-branch ablations for
`basic_spa_spe_mamba` and a single runner that can execute the complete
Indian Pines experiment matrix. The spectral branch remains the current
continuous-band tokenizer plus hidden-state BFS implementation. The runner
prepares, executes, resumes, and summarizes experiments, but implementation
of this design does not itself start the 24 full training runs.

Success means:

- S0-S7 are selected by complete, committed JSON configurations.
- All variants share one model implementation rather than copied model files.
- Direction-count changes do not silently introduce separate SSM parameters.
- One command runs 8 stages by 3 fixed seeds under the 10% Indian Pines
  protocol.
- Interrupted matrices resume without rerunning completed stage/seed jobs.
- Per-run and aggregate OA, AA, Kappa, and class accuracies are machine
  readable.

## Fixed Experimental Protocol

- Dataset: Indian Pines (`ip`)
- Training ratio: `0.1`
- Patch size: `7`
- Epochs: `300`
- Batch size: `64`
- Seeds: `1`, `34`, `20031025`
- Optimizer and scheduler: unchanged from
  `config/basic_spa_spe_mamba_ip.json`
- Validation protocol: unchanged (`val_gt = test_gt`)
- Spectral branch: unchanged continuous physical-band grouping, spectral BFS,
  and global forward/backward readout
- Formal matrix size: 8 stages x 3 seeds = 24 jobs

The runner passes the ratio and seeds explicitly. It does not depend on the
defaults in `main.py`.

## Architecture

### Single configurable model

`models/basic_spa_spe_mamba.py` remains the only model implementation. Its
spatial module accepts the following model configuration fields:

```json
{
  "spa_scan": "snake_single",
  "spa_shared_ssm": true,
  "spa_center_readout": false,
  "spa_boundary_injection": false,
  "spa_context_params": false,
  "spa_state_fusion": "none",
  "spa_state_compress_dim": 32
}
```

Supported values are deliberately narrow:

- `spa_scan`: `snake_single`, `snake_bidir`, `cross4`, or `ring_bidir`
- `spa_shared_ssm`: must be `true` for this experiment matrix
- `spa_center_readout`: boolean
- `spa_boundary_injection`: boolean
- `spa_context_params`: boolean; when enabled it controls all of `delta`,
  `B`, and `C`
- `spa_state_fusion`: `none` or `compressed`
- `spa_state_compress_dim`: positive integer, default `32`

Invalid combinations fail during model construction with a descriptive
`ValueError`. In particular, compressed state fusion is initially supported
only for `ring_bidir` with shared SSM parameters.

### Direction sharing

All spatial scan directions use one SSM parameter set. Direction sequences
are folded into the batch dimension, processed by the same one-direction
Mamba core, restored to `[B, K, L, D]`, and aligned to the original spatial
coordinates. This prevents S1-S3 from conflating scan topology with a large
direction-specific parameter increase.

For multi-path variants, aligned direction outputs are combined by global
learnable logits followed by a softmax. Logits initialize to zero, giving
uniform initial weights. S0 has no direction logits and returns its only path
directly, preserving the current baseline parameter count.

## Spatial Variants

Eight complete configurations live under `config/spa_ablation/`.

| Stage | Scan | Center readout | Boundary injection | 2D parameter context | State fusion |
|---|---|---:|---:|---:|---|
| S0 | `snake_single` | no | no | no | `none` |
| S1 | `snake_bidir` | no | no | no | `none` |
| S2 | `cross4` | no | no | no | `none` |
| S3 | `ring_bidir` | no | no | no | `none` |
| S4 | `ring_bidir` | yes | no | no | `none` |
| S5 | `ring_bidir` | no | yes | no | `none` |
| S6 | `ring_bidir` | no | yes | yes | `none` |
| S7 | `ring_bidir` | no | yes | yes | `compressed` |

S4 and S5 are independent extensions of S3. S6 extends S5. S7 extends S6.

### Scan paths

Every path is represented by a permutation of the flattened `7 x 7` grid and
an inverse permutation. Outputs are inverse-permuted before directions are
combined.

- `snake_single`: the existing row-wise S-shaped traversal.
- `snake_bidir`: the existing traversal and its exact reverse.
- `cross4`: row-wise snake, its reverse, column-wise snake, and its reverse.
- `ring_bidir`: a continuous outside-in spiral ending at the center and its
  exact reverse. Consecutive entries in both orders are four-neighbor adjacent.

Path generation supports any positive odd patch size, although formal
experiments use size 7. Tests verify uniqueness, range, reversibility, inverse
alignment, center placement, and required adjacency.

### Center-aware readout

S4 replaces global mean pooling with:

```text
z = sigmoid(beta) * x_center + (1 - sigmoid(beta)) * mean(X)
```

`beta` is a learned scalar initialized to zero, so the initial center and mean
weights are both 0.5. This affects classifier readout only; it does not alter
the spectral branch or branch fusion inside a block.

### Boundary-aware state injection

S5-S7 use the continuous physical-band tokens already computed for the
spectral branch as an HSI-specific spatial reliability signal. For position
`i` and center `c`, the reliability network receives:

```text
[t_i, t_c, abs(t_i - t_c), cosine(t_i, t_c)]
```

A two-layer MLP with hidden width 32 and GELU activation produces:

```text
r_i = 0.1 + 0.9 * sigmoid(MLP(features_i))
```

Cosine similarity uses an epsilon of `1e-6`. The center reliability is set
exactly to one. Reliability scales the generated SSM input matrix `B_i` before
selective scan. It does not scale `u_i` or the skip term `D * u_i`, so boundary
pixels retain a direct observation path.

### Two-dimensional parameter context

S6-S7 preserve the ordinary projected content as the recurrent input `u`.
They separately build a 2D context feature:

```text
q = projected + DWConv3x3(projected)
```

The context feature is gathered using the chosen scan path and generates all
three selective parameters `delta`, `B`, and `C`. Thus 2D neighborhoods decide
how information is remembered and read without replacing the content being
modeled. The existing dynamic spatial positional convolution remains present
for every stage so S6 isolates parameter conditioning rather than removing a
baseline component.

### Compressed hidden-state fusion

Only S7 uses an explicit differentiable state recurrence. The two ring-path
states are inverse-permuted to common spatial coordinates and flattened across
`d_inner * d_state`. Shared projections implement:

```text
H_down_k = P_down(H_k)
H_fused = P_up(sum_k softmax(direction_logits)_k * H_down_k)
```

`P_down` maps to `spa_state_compress_dim=32`; `P_up` maps back to the flattened
state width. The fused state is then used by each path's aligned `C` and `D*u`
observation mapping. Fusion therefore occurs before observation, not after
Mamba outputs. The two observed path outputs are aligned and combined using
the same direction softmax weights used for compressed-state aggregation.
S0-S6 continue to use the CUDA selective-scan backend.

## Configuration Files

The following complete files are added:

```text
config/spa_ablation/basic_spa_spe_mamba_s0_ip.json
config/spa_ablation/basic_spa_spe_mamba_s1_ip.json
config/spa_ablation/basic_spa_spe_mamba_s2_ip.json
config/spa_ablation/basic_spa_spe_mamba_s3_ip.json
config/spa_ablation/basic_spa_spe_mamba_s4_ip.json
config/spa_ablation/basic_spa_spe_mamba_s5_ip.json
config/spa_ablation/basic_spa_spe_mamba_s6_ip.json
config/spa_ablation/basic_spa_spe_mamba_s7_ip.json
```

Each file contains the full model and schedule configuration. Configurations
do not inherit from one another, so a saved file is sufficient to reconstruct
its stage.

## Training Entry-Point Changes

`main.py` gains optional arguments while preserving existing invocations:

- `--config PATH`: explicit JSON configuration path
- `--seed INTEGER`: run exactly one explicit seed
- `--run-name NAME`: stable experiment label used in logs
- `--output-dir PATH`: place the checkpoint and run artifacts at a requested
  location
- `--metrics-out PATH`: serialize final metrics as JSON
- `--skip-map`: do not generate the classification PNG

When `--seed` is supplied, `--num_run` must be one or omitted. The serialized
metrics include stage/run name, seed, configuration path, ratio, epochs,
parameter count, OA, AA, Kappa, per-class accuracy, confusion matrix, and best
checkpoint path. Existing default training remains available when none of the
new arguments are used.

## Unified Runner

`scripts/run_spa_ablation.py` is the sole matrix runner. Its default behavior
is equivalent to:

```bash
python scripts/run_spa_ablation.py \
  --device 0 \
  --ratio 0.1 \
  --epochs 300 \
  --batch-size 64 \
  --seeds 1 34 20031025 \
  --stages S0 S1 S2 S3 S4 S5 S6 S7
```

Each stage/seed is a separate `main.py` subprocess. This isolates CUDA memory,
random state, log handlers, and model objects between jobs.

The default artifact root is:

```text
experiments/spa_ablation_ip_10pct/
```

It contains:

```text
manifest.json
S0/seed_1/train.log
S0/seed_1/metrics.json
S0/seed_1/checkpoint/model_best.pth
...
runs.csv
summary.csv
summary.json
```

The runner writes a manifest before launching work. A job is complete only
when its `metrics.json` parses successfully and contains the expected stage,
seed, and protocol. Valid completed jobs are skipped on rerun. A missing,
invalid, or mismatched metrics file causes that job to run again.

The runner is fail-fast: a nonzero child exit records the failure in the
manifest and stops the matrix. Running the same command resumes at the first
incomplete job. It does not delete partial checkpoints or logs. `--dry-run`
validates configurations and prints the planned commands without training.

`runs.csv` contains one row per completed job. `summary.csv` and
`summary.json` group by stage and report mean and population standard deviation
for OA, AA, Kappa, parameter count, and all class accuracies.

## Validation Strategy

### CPU tests

- All path generators return valid permutations and inverse permutations.
- Bidirectional paths are exact reverses.
- Cross4 includes correct row and column traversals.
- Ring paths are four-neighbor continuous and place the center at the required
  endpoint.
- S0-S7 JSON files map to the exact table in this document.
- Invalid option combinations fail early.
- Center readout initializes to an equal mixture and has the expected shape.
- Boundary reliability is in `[0.1, 1]` and is exactly one at the center.
- Runner command generation uses the fixed protocol and exact seeds.
- Dry-run launches no training process.
- Valid metrics are skipped; invalid metrics are rerun.
- Aggregation produces correct means and population standard deviations from
  synthetic metrics.

### CUDA smoke tests

When the selective-scan extension and CUDA are available:

- Instantiate every S0-S7 configuration.
- Run forward and backward on a small batch.
- Verify classifier shape `[B, 16]`.
- Verify finite outputs, loss, and gradients.
- Verify gradients reach every enabled new module.
- Verify S0 parameter count and spatial data flow remain consistent with the
  current baseline.

CUDA-dependent tests are marked and skipped with an explicit reason when the
backend is unavailable; they are not silently treated as CPU tests.

## Error Handling and Reproducibility

- Configuration paths and allowed stage names are validated before the first
  training job.
- Seed and stage identity are written into every metrics file and manifest
  entry.
- Metrics writes use a temporary file followed by an atomic rename so an
  interrupted write cannot look complete.
- Child stdout and stderr are captured together in that job's `train.log` and
  also streamed to the runner console.
- The runner records the command, start/end time, return code, configuration
  path, and metrics path for every job.
- Existing user-generated logs, checkpoints, images, and cache files outside
  the experiment root are not modified.

## Non-Goals

- Do not run the 24 full jobs during implementation.
- Do not change the spectral tokenizer, spectral BFS, optimizer, scheduler,
  dataset split behavior, or validation/test protocol.
- Do not add direction-specific spatial SSM parameters.
- Do not tune per-stage hyperparameters.
- Do not claim S0-S7 performance before the formal matrix completes.
- Do not merge, push, or delete existing user experiment artifacts as part of
  implementation.
