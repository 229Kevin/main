"""Run and aggregate the fixed S0-S7 Indian Pines spatial ablation."""

import argparse
import csv
import datetime
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = tuple(f"S{index}" for index in range(8))
DEFAULT_SEEDS = (1, 34, 20031025)
EXPECTED_FLAGS = {
    "S0": ("snake_single", False, False, False, "none"),
    "S1": ("snake_bidir", False, False, False, "none"),
    "S2": ("cross4", False, False, False, "none"),
    "S3": ("ring_bidir", False, False, False, "none"),
    "S4": ("ring_bidir", True, False, False, "none"),
    "S5": ("ring_bidir", False, True, False, "none"),
    "S6": ("ring_bidir", False, True, True, "none"),
    "S7": ("ring_bidir", False, True, True, "compressed"),
}


@dataclass(frozen=True)
class Job:
    stage: str
    seed: int
    config_path: Path
    output_dir: Path
    metrics_path: Path
    command: tuple[str, ...]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dataset-name", default="ip")
    parser.add_argument("--dataset-dir", default="./datasets")
    parser.add_argument("--root", type=Path, default=Path("experiments/spa_ablation_ip_10pct"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_config(stage, path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {stage} config: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        model = config["model"]
        actual = (
            model["spa_scan"],
            model["spa_center_readout"],
            model["spa_boundary_injection"],
            model["spa_context_params"],
            model["spa_state_fusion"],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid {stage} config {path}: {error}") from error
    if actual != EXPECTED_FLAGS[stage]:
        raise ValueError(f"Config {path} does not match the {stage} ablation")
    if model.get("spa_shared_ssm") is not True:
        raise ValueError(f"Config {path} must enable spa_shared_ssm")


def build_jobs(opts):
    stages = [stage.upper() for stage in opts.stages]
    if len(stages) != len(set(stages)):
        raise ValueError("Duplicate stages are not allowed")
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}")
    if len(opts.seeds) != len(set(opts.seeds)):
        raise ValueError("Duplicate seeds are not allowed")
    if not 0 < opts.ratio < 1:
        raise ValueError("--ratio must be between zero and one")
    if opts.epochs < 1 or opts.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    experiment_root = opts.root if opts.root.is_absolute() else REPO_ROOT / opts.root

    jobs = []
    for stage in stages:
        config_path = REPO_ROOT / "config" / "spa_ablation" / (
            f"basic_spa_spe_mamba_{stage.lower()}_ip.json"
        )
        _validate_config(stage, config_path)
        for seed in opts.seeds:
            output_dir = experiment_root / stage / f"seed_{seed}"
            metrics_path = output_dir / "metrics.json"
            run_name = f"{stage}_seed{seed}"
            command = (
                str(opts.python),
                str(REPO_ROOT / "main.py"),
                "--model", "basic_spa_spe_mamba",
                "--dataset_name", str(opts.dataset_name),
                "--dataset_dir", str(opts.dataset_dir),
                "--device", str(opts.device),
                "--config", str(config_path),
                "--seed", str(seed),
                "--num_run", "1",
                "--ratio", str(opts.ratio),
                "--epoch", str(opts.epochs),
                "--bs", str(opts.batch_size),
                "--run-name", run_name,
                "--output-dir", str(output_dir),
                "--metrics-out", str(metrics_path),
                "--skip-map",
            )
            jobs.append(Job(stage, seed, config_path, output_dir, metrics_path, command))
    return jobs


def _read_metrics(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def metrics_complete(job, ratio):
    payload = _read_metrics(job.metrics_path)
    if payload is None:
        return False
    required = ("oa", "aa", "kappa", "class_accuracy")
    if payload.get("run_name") != f"{job.stage}_seed{job.seed}":
        return False
    if payload.get("seed") != job.seed or payload.get("ratio") != ratio:
        return False
    if any(key not in payload for key in required):
        return False
    if not isinstance(payload["class_accuracy"], list) or not payload["class_accuracy"]:
        return False
    numeric = [payload[key] for key in ("oa", "aa", "kappa")]
    numeric.extend(payload["class_accuracy"])
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric)


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_job(job, ratio):
    if metrics_complete(job, ratio):
        print(f"[skip] {job.stage} seed={job.seed}")
        return _read_metrics(job.metrics_path)

    job.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job.output_dir / "manifest.json"
    manifest = {
        "stage": job.stage,
        "seed": job.seed,
        "config": str(job.config_path),
        "command": list(job.command),
        "started_at": _utc_now(),
        "status": "running",
    }
    _atomic_write_json(manifest_path, manifest)
    process = subprocess.Popen(
        job.command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
    return_code = process.wait()
    manifest["return_code"] = return_code
    manifest["finished_at"] = _utc_now()
    if return_code != 0:
        manifest["status"] = "failed"
        _atomic_write_json(manifest_path, manifest)
        raise RuntimeError(f"{job.stage} seed={job.seed} exited with {return_code}")
    if not metrics_complete(job, ratio):
        manifest["status"] = "invalid_metrics"
        _atomic_write_json(manifest_path, manifest)
        raise RuntimeError(f"{job.stage} seed={job.seed} did not produce valid metrics")
    payload = _read_metrics(job.metrics_path)
    manifest["status"] = "complete"
    manifest["metrics"] = payload
    _atomic_write_json(manifest_path, manifest)
    return payload


def _metric_summary(values):
    values = [float(value) for value in values]
    return {"mean": statistics.fmean(values), "std": statistics.pstdev(values)}


def aggregate_results(jobs, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for job in jobs:
        payload = _read_metrics(job.metrics_path)
        if payload is None:
            raise ValueError(f"Missing metrics for {job.stage} seed={job.seed}")
        records.append({"stage": job.stage, **payload})

    max_classes = max(len(record["class_accuracy"]) for record in records)
    run_fields = [
        "stage", "run_name", "seed", "ratio", "oa", "aa", "kappa",
        "parameter_count",
    ] + [f"class_{index}_accuracy" for index in range(max_classes)]
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=run_fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in run_fields}
            for index, value in enumerate(record["class_accuracy"]):
                row[f"class_{index}_accuracy"] = value
            writer.writerow(row)

    summary = {}
    for stage in STAGES:
        stage_records = [record for record in records if record["stage"] == stage]
        if not stage_records:
            continue
        stage_summary = {
            metric: _metric_summary(record[metric] for record in stage_records)
            for metric in ("oa", "aa", "kappa", "parameter_count")
        }
        stage_summary["class_accuracy"] = [
            _metric_summary(record["class_accuracy"][index] for record in stage_records)
            for index in range(len(stage_records[0]["class_accuracy"]))
        ]
        summary[stage] = stage_summary

    _atomic_write_json(root / "summary.json", summary)
    summary_fields = ["stage"]
    for metric in ("oa", "aa", "kappa", "parameter_count"):
        summary_fields.extend((f"{metric}_mean", f"{metric}_std"))
    for index in range(max_classes):
        summary_fields.extend((f"class_{index}_accuracy_mean", f"class_{index}_accuracy_std"))
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary_fields)
        writer.writeheader()
        for stage, values in summary.items():
            row = {"stage": stage}
            for metric in ("oa", "aa", "kappa", "parameter_count"):
                row[f"{metric}_mean"] = values[metric]["mean"]
                row[f"{metric}_std"] = values[metric]["std"]
            for index, class_values in enumerate(values["class_accuracy"]):
                row[f"class_{index}_accuracy_mean"] = class_values["mean"]
                row[f"class_{index}_accuracy_std"] = class_values["std"]
            writer.writerow(row)
    return summary


def main(argv=None):
    opts = parse_args(argv)
    jobs = build_jobs(opts)
    if opts.dry_run:
        for job in jobs:
            print(shlex.join(job.command))
        return 0
    for job in jobs:
        run_job(job, opts.ratio)
    aggregate_results(jobs, jobs[0].output_dir.parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
