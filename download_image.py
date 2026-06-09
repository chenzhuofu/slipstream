#!/usr/bin/env python3
"""Pre-download SWE-bench Docker images as Apptainer .sif files for offline use.

Run this on a login node (with network access) before submitting SLURM jobs
to compute nodes that have no network.

Usage:
    python download_image.py --subset verified --split test --slice 0:64 --filter "(pytest|pylint).*"
"""

import concurrent.futures
import os
import re
import shutil
import subprocess
from pathlib import Path

import typer
from datasets import load_dataset

DEFAULT_IMAGE_DIR = "/path/to/apptainer/sifs"  # set me, or pass --image-dir

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


def get_swebench_docker_image_name(instance: dict) -> tuple[str, str]:
    """Returns (docker_pull_url, local_filename_stem)."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        repo_tag = f"sweb.eval.x86_64.{id_docker_compatible}"
        full_docker_url = f"docker://docker.io/swebench/{repo_tag}:latest"
    else:
        repo_tag = image_name.replace(":", "_").replace("/", "_")
        full_docker_url = f"docker://{image_name}"
    return full_docker_url, repo_tag


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    import random

    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    if filter_spec:
        instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
    return instances


def pull_single_image(docker_url: str, sif_path: Path):
    """Executes apptainer pull with isolated cache to avoid race conditions."""
    if sif_path.exists():
        print(f"[Exists] {sif_path.name}")
        return

    print(f"[Pulling] {docker_url} -> {sif_path.name} ...")

    unique_id = sif_path.stem
    base_dir = sif_path.parent
    task_tmp_dir = base_dir / "tmp" / unique_id
    task_cache_dir = base_dir / "cache" / unique_id
    task_tmp_dir.mkdir(parents=True, exist_ok=True)
    task_cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["APPTAINER_TMPDIR"] = str(task_tmp_dir)
    env["APPTAINER_CACHEDIR"] = str(task_cache_dir)

    try:
        subprocess.run(
            ["apptainer", "pull", str(sif_path), docker_url],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        print(f"[Done] {sif_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"[Failed] {sif_path.name}\nError: {e.stderr.decode()}")
    finally:
        shutil.rmtree(task_tmp_dir, ignore_errors=True)
        shutil.rmtree(task_cache_dir, ignore_errors=True)


@app.command()
def main(
    subset: str = typer.Option("verified", "--subset", help="SWEBench subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:32')"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    workers: int = typer.Option(8, "--workers", help="Number of parallel downloads"),
    image_dir: Path = typer.Option(Path(DEFAULT_IMAGE_DIR), "--image-dir", help="Directory to save .sif files"),
):
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Images will be saved to: {image_dir}")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path} ({split})...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec)
    print(f"Found {len(instances)} instances matching criteria.")

    # Deduplicate — many instances may share the same image
    tasks = {}
    for instance in instances:
        docker_url, filename_stem = get_swebench_docker_image_name(instance)
        sif_path = image_dir / f"{filename_stem}.sif"
        tasks[sif_path] = docker_url

    print(f"Need to prepare {len(tasks)} unique images.")
    print(f"Starting downloads with {workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(pull_single_image, url, path) for path, url in tasks.items()]
        concurrent.futures.wait(futures)

    print("\nAll downloads finished.")


if __name__ == "__main__":
    app()
