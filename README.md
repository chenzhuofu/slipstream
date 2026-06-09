# Slipstream

**Asynchronous, judge-validated trajectory compaction for long-horizon LLM agents.**

> Anonymous artifact accompanying the Slipstream paper (under double-blind review).
> See [`README_UPSTREAM.md`](README_UPSTREAM.md) for the upstream `mini-swe-agent` docs this work extends.

## What is Slipstream?

Long-horizon LLM agents accumulate large contexts, so modern frameworks rely on **compaction** — an LLM rewriting the trajectory into a shorter summary the agent resumes from. Today's compaction is **synchronous on the critical path**: it costs latency and can silently degrade accuracy, because the compactor cannot know what the agent will need next, and post-compaction errors propagate as coherent but incorrect behavior.

Slipstream's insight: run the compactor **in parallel** with continued agent execution on the original context. The candidate summary and the agent's next steps are then generated independently from the same pre-compaction state, giving an independent **judge** signal that accepts the summary only if it preserves the agent's forward intent and the facts it depends on.

Across SWE-bench Verified (coding) and BrowseComp (browsing): **+8.8 pp accuracy** and **−39.7% end-to-end latency**.

## Setup

```bash
uv venv
uv pip install -e ".[dev,eval]"
```

Hardcoded paths in the analysis scripts are placeholders (`PROJECT_ROOT = Path('/path/to/project')`) — set them to your checkout before running.

For HPC grading (no Docker), pre-download Apptainer images once:

```bash
uv run download_image.py --subset verified --split test --image-dir /path/to/sif/cache
```

## Reproducing the paper

**1 — Run.** [`scripts/`](scripts/) contains SLURM launchers for the sync-vs-async sweep across 4k / 6k / 8k contexts, organized by model. `_async` variants run Slipstream; `_sync` variants are the synchronous-compaction baseline.

- [`scripts/Qwen3.5-9B/`](scripts/Qwen3.5-9B/) — Qwen3.5-9B, 1×GPU, TP=1.
- [`scripts/Seed-OSS-36B/`](scripts/Seed-OSS-36B/) — ByteDance-Seed/Seed-OSS-36B-Instruct, 2×GPU, TP=2.

Each model directory has 6 launchers: `eval_swe_{sync,async}_{4,6,8}k.slurm`. Each spawns a vLLM server, waits for it, and runs `mini-extra swebench` over SWE-bench Verified, emitting `stats.json` per instance into `results/<model>/eval_swe_<run>/<inst>/`. Set `PROJECT_ROOT` and adjust the `#SBATCH --partition` and CUDA module load to your cluster before submitting.

**2 — Grade.** [`grade.sh`](grade.sh) wraps [`eval_apptainer.py`](eval_apptainer.py):

```bash
PROJECT_ROOT=/path/to/this/repo \
APPTAINER_SIF_DIR=/path/to/sif/cache \
bash grade.sh
```

**3 — Aggregate / plot.**

- [`parse_statistics_to_excel.py`](parse_statistics_to_excel.py) → paper-table spreadsheet.

BrowseComp uses the same schema and pipeline; its launch side is a separate harness not bundled here yet.
