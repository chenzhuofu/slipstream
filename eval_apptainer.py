#!/usr/bin/env python3
"""Evaluate SWE-bench predictions using Apptainer instead of Docker.

Replicates the core logic of swebench.harness.run_evaluation but uses
apptainer sandboxes for container execution, enabling evaluation on HPC
clusters where Docker is not available.

Requires: pip install swebench

Usage (on a login node or compute node):
    python eval_apptainer.py \
        --predictions_path ./results/eval/preds.json \
        --subset verified --split test \
        --sif_dir /path/to/apptainer/sifs \
        --max_workers 8 \
        --run_id eval_run_01
"""

import concurrent.futures
import json
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

import typer
from datasets import load_dataset
from tqdm.auto import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_WORKDIR,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    PASS_TO_PASS,
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec

app = typer.Typer(add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "rebench": "nebius/SWE-rebench",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def get_sif_path(instance_id: str, sif_dir: str) -> Path:
    """Get the .sif file path for a given instance."""
    id_docker_compatible = instance_id.replace("__", "_1776_")
    return Path(sif_dir) / f"sweb.eval.x86_64.{id_docker_compatible}.sif"


def exec_in_sandbox(sandbox_dir: Path, command: str, timeout: int = 1800, binds: list[str] | None = None) -> tuple[str, int]:
    """Execute a command inside an apptainer sandbox."""
    cmd = [
        "apptainer", "--quiet", "exec",
        "--contain", "--cleanenv", "--fakeroot",
        "--pwd", DOCKER_WORKDIR,
        "--writable",
    ]
    for bind in (binds or []):
        cmd += ["-B", bind]
    cmd += [str(sandbox_dir), "bash", "-c", command]
    try:
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired as e:
        output = e.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return output, -1


def build_sandbox(sif_path: Path, max_retries: int = 3) -> Path:
    """Build a writable sandbox from a .sif file."""
    for attempt in range(max_retries):
        sandbox_dir = Path(tempfile.gettempdir()) / f"swebench-eval-{uuid.uuid4().hex[:8]}"
        try:
            subprocess.run(
                ["apptainer", "build", "--sandbox", str(sandbox_dir), str(sif_path)],
                check=True,
                capture_output=True,
            )
            return sandbox_dir
        except subprocess.CalledProcessError as e:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to build sandbox after {max_retries} attempts: {e.stderr.decode()}"
                )
    raise RuntimeError("Unreachable")


def run_instance(
    test_spec: TestSpec,
    pred: dict,
    sif_dir: str,
    log_dir: Path,
    timeout: int,
) -> dict:
    """Run evaluation for a single instance using apptainer."""
    instance_id = test_spec.instance_id

    # Check for existing report
    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {"completed": True, "resolved": report[instance_id]["resolved"]}

    log_dir.mkdir(parents=True, exist_ok=True)

    sif_path = get_sif_path(instance_id, sif_dir)
    if not sif_path.exists():
        # Try with the image_name from the dataset
        print(f"  [SKIP] No .sif file for {instance_id} at {sif_path}")
        return {"completed": False, "resolved": False}

    sandbox_dir = None
    eval_completed = False
    report = {}

    try:
        # Build sandbox
        sandbox_dir = build_sandbox(sif_path)

        # Write patch file to log dir
        patch_content = pred.get(KEY_PREDICTION) or ""
        patch_file = log_dir / "patch.diff"
        patch_file.write_text(patch_content)

        # Apply patch — bind-mount the patch file to DOCKER_PATCH so --contain doesn't hide it
        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            output, rc = exec_in_sandbox(
                sandbox_dir, f"{git_apply_cmd} {DOCKER_PATCH}", timeout=60,
                binds=[f"{patch_file}:{DOCKER_PATCH}"],
            )
            if rc == 0:
                print(f"  [{instance_id}] {APPLY_PATCH_PASS}")
                applied_patch = True
                break

        if not applied_patch:
            print(f"  [{instance_id}] {APPLY_PATCH_FAIL}")
            (log_dir / LOG_TEST_OUTPUT).write_text(f"{APPLY_PATCH_FAIL}\n{output}")
            return {"completed": False, "resolved": False}

        # Write and run eval script
        eval_script = test_spec.eval_script
        eval_script_path = sandbox_dir / "eval.sh"
        eval_script_path.write_text(eval_script)

        test_output, rc = exec_in_sandbox(sandbox_dir, "/bin/bash /eval.sh", timeout=timeout)

        # Save test output
        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output_path.write_text(test_output)
        if rc == -1:
            test_output_path.write_text(test_output + f"\n\nTimeout error: {timeout} seconds exceeded.")
            print(f"  [{instance_id}] Timed out after {timeout}s")
            return {"completed": False, "resolved": False}

        # Grade
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        resolved = report.get(instance_id, {}).get("resolved", False)
        report_path.write_text(json.dumps(report, indent=4))
        eval_completed = True
        print(f"  [{instance_id}] resolved={resolved}")

    except Exception as e:
        print(f"  [{instance_id}] Error: {e}")
        traceback.print_exc()
    finally:
        if sandbox_dir and sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    return {
        "completed": eval_completed,
        "resolved": report.get(instance_id, {}).get("resolved", False),
    }


@app.command()
def main(
    predictions_path: Path = typer.Option(..., "-p", "--predictions-path", help="Path to preds.json"),
    subset: str = typer.Option("verified", "--subset", help="SWEBench subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    sif_dir: str = typer.Option(
        "/path/to/apptainer/sifs",
        "--sif-dir",
        help="Directory with pre-downloaded .sif files",
    ),
    max_workers: int = typer.Option(4, "-w", "--max-workers", help="Parallel workers"),
    run_id: str = typer.Option("eval", "--run-id", help="Run identifier for logs"),
    timeout: int = typer.Option(1800, "-t", "--timeout", help="Timeout per instance in seconds"),
    output_dir: Path = typer.Option(None, "-o", "--output-dir", help="Output directory (default: same as predictions)"),
):
    # Load predictions
    preds_data = json.loads(predictions_path.read_text())
    predictions = {}
    for instance_id, pred in preds_data.items():
        predictions[instance_id] = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: pred.get("model_name_or_path", "unknown"),
            KEY_PREDICTION: pred.get("model_patch", ""),
        }

    # Filter out empty patches
    predictions = {k: v for k, v in predictions.items() if v[KEY_PREDICTION]}
    print(f"Loaded {len(predictions)} non-empty predictions")

    # Load dataset
    dataset_path = DATASET_MAPPING.get(subset, subset)
    dataset = list(load_dataset(dataset_path, split=split))
    dataset = [inst for inst in dataset if inst[KEY_INSTANCE_ID] in predictions]
    print(f"Matched {len(dataset)} instances from dataset")

    # Create TestSpecs
    test_specs = [make_test_spec(inst, namespace="swebench") for inst in dataset]

    # Set up output directory
    out_dir = output_dir or predictions_path.parent
    log_base = out_dir / "eval_logs" / run_id

    # Run evaluations
    stats = {"resolved": 0, "unresolved": 0, "error": 0}
    lock = threading.Lock()

    def process_one(test_spec: TestSpec) -> dict:
        iid = test_spec.instance_id
        instance_log_dir = log_base / iid
        result = run_instance(test_spec, predictions[iid], sif_dir, instance_log_dir, timeout)
        with lock:
            if result["completed"]:
                if result["resolved"]:
                    stats["resolved"] += 1
                else:
                    stats["unresolved"] += 1
            else:
                stats["error"] += 1
        return result

    print(f"Running evaluation with {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, ts): ts.instance_id for ts in test_specs}
        pbar = tqdm(total=len(futures), desc="Evaluation")
        for future in concurrent.futures.as_completed(futures):
            pbar.set_postfix(stats)
            pbar.update()
        pbar.close()

    # Summary
    total = stats["resolved"] + stats["unresolved"] + stats["error"]
    print(f"\n{'='*60}")
    print(f"Results: {stats['resolved']}/{total} resolved ({100*stats['resolved']/max(total,1):.1f}%)")
    print(f"  Resolved:   {stats['resolved']}")
    print(f"  Unresolved: {stats['unresolved']}")
    print(f"  Errors:     {stats['error']}")
    print(f"{'='*60}")

    # Save summary
    summary_path = out_dir / f"eval_summary_{run_id}.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    app()
