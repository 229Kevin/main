# S0-S7 Spatial Mamba Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build one configurable spatial Mamba, eight complete S0-S7 configurations, and a resumable runner for the 24-job Indian Pines 10% ablation matrix.

**Architecture:** Keep basic_spa_spe_mamba as the only model entry point. Refactor its Spa branch into shared-parameter multi-path scanning with optional center readout, boundary-aware state injection, 2D parameter context, and compressed pre-observation state fusion. Extend main.py for one reproducible job and orchestrate isolated jobs with a standard-library runner.

**Tech Stack:** Python, PyTorch 2.1.2/CUDA 12.1 in LDMT-Mamba, SelectiveScanOflex, NumPy, argparse, json, csv, subprocess, unittest.

**Spec:** docs/superpowers/specs/2026-09-20-spa-ablation-design.md

## Global Constraints

- Protocol: Indian Pines, ratio 0.1, patch 7, epochs 300, batch 64, seeds 1, 34, 20031025.
- Preserve continuous-band tokenization, spectral BFS, optimizer, scheduler, split behavior, and val_gt = test_gt.
- Spatial paths share one SSM parameter set. S0-S6 use CUDA scan; only S7 uses explicit spatial recurrence.
- S4 and S5 independently extend S3; S6 extends S5; S7 extends S6.
- Do not run 24 formal jobs or touch existing user experiment artifacts during implementation.
- Run tests with /home/zwk/anaconda3/envs/LDMT-Mamba/bin/python and standard-library unittest.

## Review Focus

- Invalid scan names, even patch sizes, incompatible fusion modes, and nonpositive compression widths fail early; Tasks 1 and 5 test them.
- Cross4 and ring inverse alignment is exact; Task 1 tests coordinate round trips.
- Missing new fields remains S0 with 419476 parameters; Tasks 5 and 9 enforce this.
- Corrupt or identity-mismatched metrics never count as complete; Task 8 tests each mismatch.
- Missing CUDA/extension explicitly skips CUDA tests while CPU orchestration tests still run; Task 9 owns this.

---

## File Map

- Modify models/basic_spa_spe_mamba.py: paths, shared scan, optional mechanisms, validation, integration.
- Modify main.py: import-safe CLI, single seed, output paths, atomic metrics.
- Create config/spa_ablation/basic_spa_spe_mamba_s0_ip.json through s7.
- Create scripts/__init__.py and scripts/run_spa_ablation.py.
- Create tests for paths, modules, configs, CLI, runner, and CUDA smoke.
- Modify README.md with exact commands.

### Task 1: Deterministic Spatial Paths

**Files:**
- Modify: models/basic_spa_spe_mamba.py:294-299
- Create: tests/__init__.py
- Create: tests/test_spatial_paths.py

**Interfaces:**
- Produces build_spatial_orders(scan: str, size: int) -> tuple[torch.Tensor, ...].

- [ ] **Step 1: Write failing path tests**

~~~python
import unittest
import torch
from models.basic_spa_spe_mamba import build_spatial_orders

class SpatialPathTests(unittest.TestCase):
    def test_counts_reverses_and_round_trip(self):
        self.assertEqual(len(build_spatial_orders("snake_single", 7)), 1)
        self.assertEqual(len(build_spatial_orders("cross4", 7)), 4)
        source = torch.arange(49)
        for name in ("snake_single", "snake_bidir", "cross4", "ring_bidir"):
            for order in build_spatial_orders(name, 7):
                self.assertEqual(sorted(order.tolist()), list(range(49)))
                restored = source.index_select(0, order).index_select(0, order.argsort())
                self.assertTrue(torch.equal(restored, source))

    def test_ring_continuity_and_center(self):
        outside_in, inside_out = build_spatial_orders("ring_bidir", 7)
        self.assertTrue(torch.equal(inside_out, outside_in.flip(0)))
        self.assertEqual(outside_in[-1].item(), 24)
        for left, right in zip(outside_in.tolist(), outside_in.tolist()[1:]):
            lr, lc = divmod(left, 7)
            rr, rc = divmod(right, 7)
            self.assertEqual(abs(lr - rr) + abs(lc - rc), 1)

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "Unsupported spa_scan"):
            build_spatial_orders("diagonal", 7)
        with self.assertRaisesRegex(ValueError, "positive odd"):
            build_spatial_orders("snake_single", 6)
~~~

- [ ] **Step 2: Run and confirm missing-symbol failure**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_paths -v
~~~

- [ ] **Step 3: Implement row snake, column snake, clockwise shrinking-boundary spiral, reverse paths, and descriptive errors**

~~~python
def _row_snake_order(size):
    grid = torch.arange(size * size, dtype=torch.long).view(size, size)
    grid[1::2] = grid[1::2].flip(1)
    return grid.flatten()

def _column_snake_order(size):
    return _row_snake_order(size).view(size, size).transpose(0, 1).flatten()

def build_spatial_orders(scan, size):
    if size <= 0 or size % 2 == 0:
        raise ValueError("Spatial patch size must be a positive odd integer")
    row = _row_snake_order(size)
    if scan == "snake_single": return (row,)
    if scan == "snake_bidir": return row, row.flip(0)
    if scan == "cross4":
        column = _column_snake_order(size)
        return row, row.flip(0), column, column.flip(0)
    if scan == "ring_bidir":
        ring = _outside_in_spiral_order(size)
        return ring, ring.flip(0)
    raise ValueError(f"Unsupported spa_scan: {scan}")
~~~

Keep _snake_indices as a wrapper around the first snake order and argsort.

- [ ] **Step 4: Verify and commit**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_paths -v
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m py_compile models/basic_spa_spe_mamba.py
git add models/basic_spa_spe_mamba.py tests/__init__.py tests/test_spatial_paths.py
git commit -m "feat: add spatial ablation scan paths"
~~~

### Task 2: Shared Multi-Path CUDA Scan and S0-S3

**Files:**
- Modify: models/basic_spa_spe_mamba.py:21-200,367-400
- Create: tests/test_spatial_modules.py

**Interfaces:**
- Adds EditableMambaCore.forward_shared(sequences, parameter_features=None, reliability=None).
- Adds SpatialMultiPathMamba.forward(x, spectral_tokens=None) -> Tensor[B,H,W,D].

- [ ] **Step 1: Write failing sharing, path-count, and alignment tests**

~~~python
import unittest
import torch
from models.basic_spa_spe_mamba import EditableMambaCore, SpatialMultiPathMamba

class MultiPathTests(unittest.TestCase):
    def test_counts_and_single_parameter_group(self):
        counts = {"snake_single":1, "snake_bidir":2, "cross4":4, "ring_bidir":2}
        for scan, count in counts.items():
            module = SpatialMultiPathMamba(7, 8, spa_scan=scan, d_state=4, ssm_ratio=1, d_conv=3)
            self.assertEqual(module.num_paths, count)
            self.assertEqual(module.mamba.num_directions, 1)
            self.assertEqual(module.mamba.x_proj_weight.shape[0], 1)

    def test_alignment_before_fusion(self):
        module = SpatialMultiPathMamba(7, 8, spa_scan="cross4", d_state=4, ssm_ratio=1, d_conv=3)
        source = torch.arange(49, dtype=torch.float32).view(1,7,7,1).repeat(1,1,1,8)
        module.dynamic_pos = torch.nn.Identity()
        module.norm = torch.nn.Identity()
        module.mamba.forward_shared = lambda sequences, parameter_features=None, reliability=None: sequences
        self.assertTrue(torch.equal(module(source), source))

@unittest.skipUnless(torch.cuda.is_available(), "CUDA selective scan unavailable")
class SharedCudaScanTests(unittest.TestCase):
    def test_single_path_matches_existing_forward(self):
        torch.manual_seed(7)
        core = EditableMambaCore(8, num_directions=1, d_state=4, ssm_ratio=1, d_conv=3).cuda().eval()
        x = torch.randn(2,1,9,8,device="cuda")
        self.assertTrue(torch.allclose(core(x), core.forward_shared(x), atol=1e-5, rtol=1e-5))
~~~

- [ ] **Step 2: Run and confirm missing APIs**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_modules -v
~~~

- [ ] **Step 3: Factor _project_content and implement forward_shared**

Fold K into batch; use only x_proj_weight[0], dt_proj_weight[0], dt_proj_bias[0], one A block, and one D block. parameter_features must match [B,K,L,d_inner] and supplies the joint delta/B/C projection. reliability must match [B,K,L] and scales only B. Restore [B,K,L,D] after SelectiveScanOflex. Preserve forward and spectral forward_bfs.

- [ ] **Step 4: Implement gather, inverse alignment, and fusion**

~~~python
def _gather_paths(self, feature_map):
    flat = feature_map.reshape(feature_map.shape[0], self.patch_size ** 2, feature_map.shape[-1])
    return torch.stack([flat.index_select(1, order) for order in self.orders], 1)

def _align_paths(self, paths):
    return torch.stack([paths[:, i].index_select(1, inv) for i, inv in enumerate(self.inverse_orders)], 1)

def _combine_paths(self, aligned):
    if self.num_paths == 1: return aligned[:, 0]
    return torch.einsum("k,bklc->blc", self.direction_logits.softmax(0), aligned)
~~~

Apply existing dynamic position convolution and LayerNorm before gathering. Create no direction logits for S0. Export SpatialSnakeMamba as a compatibility alias.

- [ ] **Step 5: Verify and commit**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_paths tests.test_spatial_modules -v
git add models/basic_spa_spe_mamba.py tests/test_spatial_modules.py
git commit -m "feat: add shared multi-path spatial scans"
~~~

### Task 3: Center Readout, Boundary Injection, and 2D Context

**Files:**
- Modify: models/basic_spa_spe_mamba.py:143-200,367-580
- Modify: tests/test_spatial_modules.py

**Interfaces:**
- Adds BoundaryReliability(hidden_dim, mlp_hidden_dim=32), center_mean_pool(x,beta), and _parameter_context(x).

- [ ] **Step 1: Write failing CPU tests**

~~~python
from models.basic_spa_spe_mamba import BoundaryReliability, center_mean_pool

class GuidanceTests(unittest.TestCase):
    def test_reliability_range_center_and_readout(self):
        r = BoundaryReliability(8)(torch.randn(3,7,7,8))
        self.assertTrue(torch.all(r >= 0.1) and torch.all(r <= 1.0))
        self.assertTrue(torch.equal(r[:,3,3], torch.ones(3)))
        x = torch.randn(2,7,7,8)
        actual = center_mean_pool(x, torch.nn.Parameter(torch.zeros(())))
        self.assertTrue(torch.equal(actual, 0.5*x[:,3,3] + 0.5*x.mean((1,2))))

    def test_context_shape(self):
        module = SpatialMultiPathMamba(7,8,spa_scan="ring_bidir",spa_context_params=True,d_state=4,ssm_ratio=1,d_conv=3)
        context = module._parameter_context(torch.randn(2,7,7,8))
        self.assertEqual(tuple(context.shape), (2,2,49,module.mamba.d_inner))
~~~

- [ ] **Step 2: Run and confirm failures**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_modules.GuidanceTests -v
~~~

- [ ] **Step 3: Implement exact guidance equations**

Boundary MLP is Linear(3D+1,32), GELU, Linear(32,1). Its input is [t_i,t_c,abs(t_i-t_c),cosine] with eps 1e-6; output is 0.1+0.9*sigmoid, with center overwritten to 1. center_mean_pool uses sigmoid(beta), beta initialized zero.

- [ ] **Step 4: Implement conditional context**

Create depthwise Conv2d(d_inner,d_inner,3,padding=1,groups=d_inner) only when enabled. Project the normalized 2D map with shared in_proj, take content half if gated, add DWConv, apply SiLU, and gather it by the same paths. It generates all delta/B/C; ordinary path projection remains u. Gather reliability with the same orders and pass it separately.

- [ ] **Step 5: Verify and commit**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_paths tests.test_spatial_modules -v
git add models/basic_spa_spe_mamba.py tests/test_spatial_modules.py
git commit -m "feat: add spatial guidance and parameter context"
~~~

### Task 4: Compressed Pre-Observation State Fusion

**Files:**
- Modify: models/basic_spa_spe_mamba.py:21-291,334-445
- Modify: tests/test_spatial_modules.py

**Interfaces:**
- Adds SharedScanInternals, scan_states_shared, observe_shared, CompressedSpatialStateFusion.

- [ ] **Step 1: Write failing shape, gradient, and invalid-mode tests**

~~~python
from models.basic_spa_spe_mamba import BasicSpaSpeMamba, CompressedSpatialStateFusion

class StateFusionTests(unittest.TestCase):
    def test_shape_and_gradients(self):
        module = CompressedSpatialStateFusion(8,4,5)
        states = torch.randn(2,2,8,4,49,requires_grad=True)
        fused = module(states, torch.tensor([0.25,0.75]))
        self.assertEqual(tuple(fused.shape), (2,8,4,49))
        fused.sum().backward()
        self.assertIsNotNone(module.down.weight.grad)
        self.assertIsNotNone(module.up.weight.grad)

    def test_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "ring_bidir"):
            BasicSpaSpeMamba(spa_scan="cross4",spa_state_fusion="compressed")
        with self.assertRaisesRegex(ValueError, "positive"):
            BasicSpaSpeMamba(spa_scan="ring_bidir",spa_state_fusion="compressed",spa_state_compress_dim=0)
        with self.assertRaisesRegex(ValueError, "spa_shared_ssm"):
            BasicSpaSpeMamba(spa_shared_ssm=False)
~~~

- [ ] **Step 2: Run and confirm failures**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_modules.StateFusionTests -v
~~~

- [ ] **Step 3: Implement explicit shared recurrence**

Return states [B,K,d_inner,d_state,L], C [B,K,d_state,L], u [B,K,d_inner,L], and optional gate. Reuse shared projections, reliability-scaled B, softplus delta, and the spectral recurrence equation. observe_shared computes aligned C*h + D*u before output norm, gate, out projection, and dropout.

- [ ] **Step 4: Implement compressed fusion**

~~~python
class CompressedSpatialStateFusion(nn.Module):
    def __init__(self, d_inner, d_state, compressed_dim):
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        width = d_inner * d_state
        self.down = nn.Conv1d(width, compressed_dim, 1)
        self.up = nn.Conv1d(compressed_dim, width, 1)
    def forward(self, states, weights):
        b,k,_,_,length = states.shape
        down = self.down(states.reshape(b*k,-1,length)).view(b,k,-1,length)
        return self.up(torch.einsum("k,bkcl->bcl", weights, down)).view(b,self.d_inner,self.d_state,length)
~~~

Inverse-align states, C, u, and gate before fusion. Use the same softmax weights for compressed aggregation and final observed output aggregation.

- [ ] **Step 5: Verify and commit**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_modules -v
git add models/basic_spa_spe_mamba.py tests/test_spatial_modules.py
git commit -m "feat: add compressed spatial state fusion"
~~~

### Task 5: Full Flags, S0 Compatibility, and CUDA Smoke

**Files:**
- Modify: models/basic_spa_spe_mamba.py:448-612
- Create: tests/test_spa_variants_cuda.py

**Interfaces:**
- BasicSpaSpeMamba gains seven explicit Spa fields; their defaults select S0.

- [ ] **Step 1: Thread exact defaults and validation**

~~~python
spa_scan="snake_single", spa_shared_ssm=True,
spa_center_readout=False, spa_boundary_injection=False,
spa_context_params=False, spa_state_fusion="none",
spa_state_compress_dim=32,
~~~

Require shared SSM, allowed scan/fusion names, positive compression, and ring_bidir for compressed mode. Pass spectral tokens to both branches. Create center_pool_beta only for S4; otherwise retain mean pooling exactly.

- [ ] **Step 2: Write exact guarded S0-S7 smoke matrix**

~~~python
STAGES = {
 "S0": {}, "S1":{"spa_scan":"snake_bidir"}, "S2":{"spa_scan":"cross4"},
 "S3":{"spa_scan":"ring_bidir"},
 "S4":{"spa_scan":"ring_bidir","spa_center_readout":True},
 "S5":{"spa_scan":"ring_bidir","spa_boundary_injection":True},
 "S6":{"spa_scan":"ring_bidir","spa_boundary_injection":True,"spa_context_params":True},
 "S7":{"spa_scan":"ring_bidir","spa_boundary_injection":True,"spa_context_params":True,"spa_state_fusion":"compressed"},
}
~~~

For each: CUDA forward [1,1,200,7,7], output [1,16], finite values, backward, gradients on enabled modules. Assert default total parameter count 419476. Probe a tiny scan in setUpClass; convert only missing CUDA/extension errors to SkipTest and re-raise implementation errors.

- [ ] **Step 3: Verify and commit**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spatial_paths tests.test_spatial_modules tests.test_spa_variants_cuda -v
git add models/basic_spa_spe_mamba.py tests/test_spa_variants_cuda.py
git commit -m "feat: integrate S0-S7 spatial variants"
~~~

### Task 6: Eight Complete Configurations

**Files:**
- Create: config/spa_ablation/basic_spa_spe_mamba_s0_ip.json through s7
- Create: tests/test_spa_ablation_configs.py

**Interfaces:**
- Each JSON contains the full current model and schedule plus all seven Spa fields.

- [ ] **Step 1: Write failing exact-matrix test**

~~~python
EXPECTED = {
 "S0":("snake_single",False,False,False,"none"),
 "S1":("snake_bidir",False,False,False,"none"),
 "S2":("cross4",False,False,False,"none"),
 "S3":("ring_bidir",False,False,False,"none"),
 "S4":("ring_bidir",True,False,False,"none"),
 "S5":("ring_bidir",False,True,False,"none"),
 "S6":("ring_bidir",False,True,True,"none"),
 "S7":("ring_bidir",False,True,True,"compressed"),
}
~~~

Load each path and compare scan, center, boundary, context, fusion. Assert shared true, compression 32, patch 7, bands 200, classes 16, AdamW lr .001, StepLR 30/.9.

- [ ] **Step 2: Run and confirm missing files**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spa_ablation_configs -v
~~~

- [ ] **Step 3: Create eight standalone files**

Copy the complete current config, insert every explicit Spa field, and use the exact matrix. Do not use inheritance or runtime overrides.

- [ ] **Step 4: Verify and commit**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_spa_ablation_configs -v
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m json.tool config/spa_ablation/basic_spa_spe_mamba_s7_ip.json >/dev/null
git add config/spa_ablation tests/test_spa_ablation_configs.py
git commit -m "config: add S0-S7 spatial ablations"
~~~

### Task 7: Reproducible Single-Job Interface

**Files:**
- Modify: main.py:34-200
- Create: tests/test_main_experiment_cli.py

**Interfaces:**
- Produces build_parser, resolve_config_path, resolve_seeds, json_ready_metrics, atomic_write_json, main(argv=None).
- Adds --config, --seed, --run-name, --output-dir, --metrics-out, --skip-map.

- [ ] **Step 1: Write failing parser and atomic JSON tests**

Test seed 34 resolves [34]; seed with num_run 3 raises; NumPy metrics serialize to oa/aa/kappa/class_accuracy/confusion_matrix; atomic write leaves no .tmp file.

- [ ] **Step 2: Run and confirm helper failures**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_main_experiment_cli -v
~~~

- [ ] **Step 3: Extract parser and seed/config resolution**

~~~python
def resolve_seeds(opts):
    if opts.seed is not None:
        if opts.num_run not in (None, 1):
            raise ValueError("--seed requires --num_run 1 or omission")
        return [opts.seed]
    defaults = {
      "ip":[1,34,20031025,9,10,11,424124,21213,123,29],
      "whulk":[9,10,11,424124,21213,231419123,123,1,34,20031025],
      "hu2018":[202203,1,34,3,202202,202203,424124,21213,123,29]}
    count = 5 if opts.num_run is None else opts.num_run
    return defaults[opts.dataset_name][:count]
~~~

Explicit config wins; otherwise legacy path. Importing main must not parse or train.

- [ ] **Step 4: Implement stable artifacts and atomic metrics**

With output dir use train.log, checkpoint/model_best.pth, and explicit metrics path. Metrics contain run name, seed, config, ratio, epochs, parameter count, OA, AA, Kappa, class accuracies, confusion matrix, checkpoint. Write after successful inference. Preserve legacy timestamp locations otherwise. Honor skip-map.

- [ ] **Step 5: Verify and commit**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_main_experiment_cli -v
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python main.py --help >/dev/null
git add main.py tests/test_main_experiment_cli.py
git commit -m "feat: add reproducible single-job output"
~~~

### Task 8: Resumable Runner and Aggregation

**Files:**
- Create: scripts/__init__.py
- Create: scripts/run_spa_ablation.py
- Create: tests/test_run_spa_ablation.py

**Interfaces:**
- Produces frozen Job, parse_args, build_jobs, metrics_complete, run_job, aggregate_results, main.

- [ ] **Step 1: Write failing matrix/resume/dry-run/statistics tests**

Assert defaults yield 24 jobs, exact stages/seeds, ratio .1, skip-map. Test corrupt JSON, missing keys, wrong stage, seed, and ratio return false; a complete matching payload returns true. OA 90/94 yields mean 92 and population std 2. Mock Popen and assert dry-run never invokes it or creates stage directories.

- [ ] **Step 2: Run and confirm missing module**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_run_spa_ablation -v
~~~

- [ ] **Step 3: Implement jobs and preflight**

~~~python
@dataclass(frozen=True)
class Job:
    stage: str
    seed: int
    config_path: Path
    output_dir: Path
    metrics_path: Path
    command: tuple[str, ...]
~~~

Defaults: device 0, ratio .1, epochs 300, batch 64, three fixed seeds, S0-S7, ip, root experiments/spa_ablation_ip_10pct. Use sys.executable main.py with all explicit arguments. Reject duplicate/unknown stages and missing/invalid configs before launching.

- [ ] **Step 4: Implement resume, manifest, and fail-fast**

Completion requires valid JSON with matching run name, seed, ratio, OA/AA/Kappa/class accuracy. Atomically record command, times, return code, config, metrics in manifest. Stream merged stdout/stderr. Stop on first child error or invalid missing metrics. Delete nothing.

- [ ] **Step 5: Implement summaries and dry-run**

Write runs.csv; use statistics.fmean/pstdev for OA, AA, Kappa, parameter count, and every class accuracy in summary.csv and summary.json. Dry-run validates and prints 24 commands but writes no manifest/directories and launches nothing.

- [ ] **Step 6: Verify and commit**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest tests.test_run_spa_ablation -v
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python scripts/run_spa_ablation.py --dry-run
git add scripts/__init__.py scripts/run_spa_ablation.py tests/test_run_spa_ablation.py
git commit -m "feat: add resumable spatial ablation runner"
~~~

### Task 9: Documentation and Final Verification

**Files:**
- Modify: README.md
- Modify: tests/test_spa_variants_cuda.py

**Interfaces:**
- Produces verified commands; no formal experiment results.

- [ ] **Step 1: Run full suite**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest discover -s tests -v
~~~

Expected: CPU PASS; CUDA PASS on RTX 3090 or explicit SKIP only for missing CUDA/extension.

- [ ] **Step 2: Run syntax, whitespace, JSON, and command-count checks**

~~~bash
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m py_compile main.py models/basic_spa_spe_mamba.py scripts/run_spa_ablation.py
git diff --check
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python scripts/run_spa_ablation.py --dry-run > /tmp/spa_ablation_dry_run.txt
test "$(grep -c -- '--seed' /tmp/spa_ablation_dry_run.txt)" -eq 24
~~~

- [ ] **Step 3: Document exact usage**

~~~bash
conda activate LDMT-Mamba
python scripts/run_spa_ablation.py --dry-run
python scripts/run_spa_ablation.py --device 0
python scripts/run_spa_ablation.py --device 0 --stages S3 S4 --seeds 1
~~~

Document 24 jobs, 10%, output root, fail-fast, rerun-to-resume, and val_gt=test_gt.

- [ ] **Step 4: Confirm user artifacts remain untouched, commit, and rerun verification**

~~~bash
git status --short
git diff --check
git add README.md tests/test_spa_variants_cuda.py
git commit -m "docs: document spatial ablation workflow"
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python -m unittest discover -s tests -v
/home/zwk/anaconda3/envs/LDMT-Mamba/bin/python scripts/run_spa_ablation.py --dry-run
~~~
