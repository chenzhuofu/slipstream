#!/usr/bin/env python3
"""Parse per-instance stats.json files under a results directory and convert to TSV
(for pasting into Excel) plus an `overall_stats.txt` summary.

This is the mini-swe-agent analogue of FoldAgent's `parse_statistics_to_excel.py`. The
data layout differs: each instance writes `<results_dir>/<inst>/<inst>.stats.json`
(produced by `SummarizingAgent.save`) and resolution scores come from
`<results_dir>/eval_logs/<tag>/<inst>/report.json` (produced by `eval_apptainer.py`).

Usage:
    uv run python parse_statistics_to_excel.py <results_dir>
    uv run python parse_statistics_to_excel.py <results_dir> --eval-tag final
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def _load_instance_stats(stats_path: Path) -> dict:
    return json.loads(stats_path.read_text())


def _pick_eval_tag(results_dir: Path, requested: str | None) -> str | None:
    eval_logs = results_dir / "eval_logs"
    if not eval_logs.exists():
        return None
    tags = [p.name for p in eval_logs.iterdir() if p.is_dir()]
    if not tags:
        return None
    if requested:
        if requested not in tags:
            raise SystemExit(f"--eval-tag {requested!r} not found under {eval_logs} (have: {tags})")
        return requested
    if "final" in tags:
        return "final"
    return sorted(tags)[-1]


def _load_scores(results_dir: Path, eval_tag: str | None) -> dict[str, float]:
    """Return {instance_id: 1.0 if resolved else 0.0}. Missing reports → 0.0."""
    if eval_tag is None:
        return {}
    scores: dict[str, float] = {}
    for report in (results_dir / "eval_logs" / eval_tag).glob("*/report.json"):
        data = json.loads(report.read_text())
        for inst_id, inst_report in data.items():
            scores[inst_id] = 1.0 if inst_report.get("resolved") else 0.0
    return scores


def parse_results_dir(results_dir: Path, eval_tag: str | None) -> tuple[list[dict], float]:
    """Return (instances, total_program_time). total_program_time is the max total_time
    across instances (there is no batch-level timing in mini-swe-agent's stats)."""
    scores = _load_scores(results_dir, eval_tag)
    stats_paths = sorted(results_dir.glob("*/*.stats.json"))

    instances: list[dict] = []
    for p in stats_paths:
        d = _load_instance_stats(p)
        steps = d.get("step_statistics", [])
        num_steps = sum(1 for s in steps if s.get("type") == "agent")

        def _type_tokens(stype: str, field: str) -> tuple[float, float]:
            vals = [s.get(field) or 0 for s in steps if s.get("type") == stype]
            return sum(vals), (sum(vals) / len(vals) if vals else 0)

        def _type_time(stype: str) -> tuple[float, float]:
            vals = [s.get("time") or 0 for s in steps if s.get("type") == stype]
            return sum(vals), (sum(vals) / len(vals) if vals else 0)

        agent_tok_sum, agent_tok_avg = _type_tokens("agent", "completion_tokens")
        summary_tok_sum, summary_tok_avg = _type_tokens("summary", "completion_tokens")
        # mini-swe-agent's action entries carry `output_tokens` (not `completion_tokens`).
        action_tok_sum, action_tok_avg = _type_tokens("action", "output_tokens")

        agent_time_sum, agent_time_avg = _type_time("agent")
        summary_time_sum, summary_time_avg = _type_time("summary")
        action_time_sum, action_time_avg = _type_time("action")
        judge_time_sum, judge_time_avg = _type_time("judge")

        total_time = d.get("total_time", 0) or 0
        _seq = agent_time_sum + action_time_sum + summary_time_sum + judge_time_sum
        per_type_stats = {
            "agent": {
                "time_sum": agent_time_sum,
                "time_avg": agent_time_avg,
                "time_percentage": (agent_time_sum / _seq * 100) if _seq > 0 else 0,
                "tokens_sum": agent_tok_sum,
                "tokens_avg": agent_tok_avg,
            },
            "run_action": {
                "time_sum": action_time_sum,
                "time_avg": action_time_avg,
                "time_percentage": (action_time_sum / _seq * 100) if _seq > 0 else 0,
                "tokens_sum": action_tok_sum,
                "tokens_avg": action_tok_avg,
            },
            "summary": {
                "time_sum": summary_time_sum,
                "time_avg": summary_time_avg,
                "time_percentage": (summary_time_sum / _seq * 100) if _seq > 0 else 0,
                "tokens_sum": summary_tok_sum,
                "tokens_avg": summary_tok_avg,
            },
            "judge": {
                "time_sum": judge_time_sum,
                "time_avg": judge_time_avg,
                "time_percentage": (judge_time_sum / _seq * 100) if _seq > 0 else 0,
            },
        }

        n_spec_accepted = d.get("n_spec_accepted", 0)
        n_spec_rejected = d.get("n_spec_rejected", 0)
        has_speculation = bool(n_spec_accepted or n_spec_rejected) or any(s.get("speculative") for s in steps)

        # total_speculative_steps = sum of carried-over (accepted) spec steps across all
        # speculative summary events on this instance.
        total_speculative_steps = 0
        for s in steps:
            if s.get("type") != "summary":
                continue
            if _summary_spec_was_rejected(s):
                continue
            total_speculative_steps += s.get("speculative_steps", 0) or 0

        # Per-instance speculation overlap: min / max of (summary_time, concurrent spec-step time),
        # aggregated across all accepted speculative summary events on this instance.
        inst_sum_time, inst_spec_time = _aggregate_overlap(steps)
        if inst_sum_time > 0 and inst_spec_time > 0:
            speculation_overlap = min(inst_sum_time, inst_spec_time) / max(inst_sum_time, inst_spec_time)
        else:
            speculation_overlap = 0.0

        # Use wall-clock total_time as-is. Per-type times don't capture between-step overhead
        # (apptainer startup, message prep, retries) and async overlaps summary with agent
        # work, so the sum-of-types is not a faithful denominator either way. Wall-clock is
        # what the user paid in elapsed time, so report that.

        inst_id = d.get("instance_id") or p.parent.name
        instances.append(
            {
                "instance_id": inst_id,
                "num_steps": num_steps,
                "score": scores.get(inst_id, 0.0),
                "total_time": total_time,
                "has_speculation": has_speculation,
                "total_speculative_steps": total_speculative_steps,
                "speculation_overlap": speculation_overlap,
                "inst_sum_time": inst_sum_time,
                "inst_spec_time": inst_spec_time,
                "n_spec_accepted": n_spec_accepted,
                "n_spec_rejected": n_spec_rejected,
                "n_summary_fallback": d.get("n_summary_fallback", 0),
                "actual_savings_from_agent": d.get("actual_savings", 0) or 0,
                "steps": steps,
                "per_type_stats": per_type_stats,
            }
        )

    total_program_time = max((i["total_time"] for i in instances), default=0)
    return instances, total_program_time


def _summary_spec_was_rejected(s: dict) -> bool:
    """True if this summary entry represents a rejected speculative summary.

    mini-swe-agent marks rejection with ``discarded=True``; also accept the FoldAgent flags
    (``rejected`` / ``spec_rejected``) for cross-compat.
    """
    return bool(s.get("discarded") or s.get("rejected") or s.get("spec_rejected"))


def _aggregate_overlap(steps: list[dict]) -> tuple[float, float]:
    """Return (total_accepted_summary_time, total_concurrent_spec_step_time) across this
    instance. Rejected spec events contribute nothing (their spec work is replayed by a sync
    fallback, so there is no real overlap)."""
    inst_sum_time, inst_spec_time = 0.0, 0.0
    for i, s in enumerate(steps):
        if s.get("type") != "summary":
            continue
        if _summary_spec_was_rejected(s):
            continue
        n_spec = s.get("speculative_steps", 0) or 0
        s_time = s.get("time")
        if s_time is None or n_spec == 0:
            continue
        spec_t, count = 0.0, 0
        j = i + 1
        while j < len(steps) and count < n_spec:
            sj = steps[j]
            if sj.get("type") == "summary":
                break
            if sj.get("type") in ("agent", "action"):
                spec_t += sj.get("time", 0) or 0
                if sj.get("type") == "agent":
                    count += 1
            j += 1
        inst_sum_time += s_time
        inst_spec_time += spec_t
    return inst_sum_time, inst_spec_time


def create_csv_files(instances: list[dict], summary_file: Path) -> None:
    fieldnames = [
        "Instance ID",
        "Num Steps",
        "Score",
        "Total Time (s)",
        "Has Speculation",
        "Speculative Steps",
        "N Spec Accepted",
        "N Spec Rejected",
        "N Summary Fallback",
        "Agent Tokens Sum",
        "Agent Tokens Avg",
        "Agent Time Sum (s)",
        "Agent Time Avg (s)",
        "Agent Time %",
        "Summary Tokens Sum",
        "Summary Tokens Avg",
        "Summary Time Sum (s)",
        "Summary Time Avg (s)",
        "Summary Time %",
        "Action Tokens Sum",
        "Action Tokens Avg",
        "Action Time Sum (s)",
        "Action Time Avg (s)",
        "Action Time %",
        "Judge Time Sum (s)",
        "Judge Time Avg (s)",
        "Judge Time %",
        "Speculation Overlap",
    ]
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for inst in instances:
            writer.writerow(
                {
                    "Instance ID": inst["instance_id"],
                    "Num Steps": inst["num_steps"],
                    "Score": inst["score"],
                    "Total Time (s)": inst["total_time"],
                    "Has Speculation": inst["has_speculation"],
                    "Speculative Steps": inst["total_speculative_steps"],
                    "N Spec Accepted": inst["n_spec_accepted"],
                    "N Spec Rejected": inst["n_spec_rejected"],
                    "N Summary Fallback": inst["n_summary_fallback"],
                    "Agent Tokens Sum": inst["per_type_stats"]["agent"]["tokens_sum"],
                    "Agent Tokens Avg": inst["per_type_stats"]["agent"]["tokens_avg"],
                    "Agent Time Sum (s)": inst["per_type_stats"]["agent"]["time_sum"],
                    "Agent Time Avg (s)": inst["per_type_stats"]["agent"]["time_avg"],
                    "Agent Time %": inst["per_type_stats"]["agent"]["time_percentage"],
                    "Summary Tokens Sum": inst["per_type_stats"]["summary"]["tokens_sum"],
                    "Summary Tokens Avg": inst["per_type_stats"]["summary"]["tokens_avg"],
                    "Summary Time Sum (s)": inst["per_type_stats"]["summary"]["time_sum"],
                    "Summary Time Avg (s)": inst["per_type_stats"]["summary"]["time_avg"],
                    "Summary Time %": inst["per_type_stats"]["summary"]["time_percentage"],
                    "Action Tokens Sum": inst["per_type_stats"]["run_action"]["tokens_sum"],
                    "Action Tokens Avg": inst["per_type_stats"]["run_action"]["tokens_avg"],
                    "Action Time Sum (s)": inst["per_type_stats"]["run_action"]["time_sum"],
                    "Action Time Avg (s)": inst["per_type_stats"]["run_action"]["time_avg"],
                    "Action Time %": inst["per_type_stats"]["run_action"]["time_percentage"],
                    "Judge Time Sum (s)": inst["per_type_stats"]["judge"]["time_sum"],
                    "Judge Time Avg (s)": inst["per_type_stats"]["judge"]["time_avg"],
                    "Judge Time %": inst["per_type_stats"]["judge"]["time_percentage"],
                    "Speculation Overlap": inst["speculation_overlap"],
                }
            )
    print(f"Created: {summary_file}")


def _calc_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "p50": 0}
    return {"mean": statistics.mean(values), "p50": statistics.median(values)}


def _collect_stats(instances: list[dict]) -> dict:
    scores = [inst["score"] for inst in instances]

    summary_ratios, judge_ratios = [], []
    num_steps_list, speculative_steps_list = [], []
    total_times, total_tokens = [], []
    agent_tokens_list, agent_times_list = [], []
    action_tokens_list, action_times_list = [], []
    summary_tokens_list, summary_times_list = [], []
    judge_times_list = []
    speculation_overlaps = []
    spec_sum_times, spec_steps_times = [], []
    n_accepted_list, n_rejected_list, n_fallback_list = [], [], []

    for inst in instances:
        agent_tokens = inst["per_type_stats"]["agent"]["tokens_sum"]
        agent_time = inst["per_type_stats"]["agent"]["time_sum"]
        action_tokens = inst["per_type_stats"]["run_action"]["tokens_sum"]
        action_time = inst["per_type_stats"]["run_action"]["time_sum"]
        summary_tokens = inst["per_type_stats"]["summary"]["tokens_sum"]
        summary_time = inst["per_type_stats"]["summary"]["time_sum"]
        judge_time = inst["per_type_stats"]["judge"]["time_sum"]

        num_steps_list.append(inst["num_steps"])
        speculative_steps_list.append(inst["total_speculative_steps"])
        total_times.append(inst["total_time"])
        agent_tokens_list.append(agent_tokens)
        agent_times_list.append(agent_time)
        action_tokens_list.append(action_tokens)
        action_times_list.append(action_time)
        summary_tokens_list.append(summary_tokens)
        summary_times_list.append(summary_time)
        judge_times_list.append(judge_time)

        seq = agent_time + action_time + summary_time + judge_time
        summary_ratios.append((summary_time / seq * 100) if seq > 0 else 0)
        judge_ratios.append((judge_time / seq * 100) if seq > 0 else 0)
        total_tokens.append(agent_tokens + action_tokens + summary_tokens)
        speculation_overlaps.append(inst["speculation_overlap"])
        spec_sum_times.append(inst["inst_sum_time"])
        spec_steps_times.append(inst["inst_spec_time"])
        n_accepted_list.append(inst["n_spec_accepted"])
        n_rejected_list.append(inst["n_spec_rejected"])
        n_fallback_list.append(inst["n_summary_fallback"])

    total_spec_events = sum(n_accepted_list) + sum(n_rejected_list)
    reject_rate = (sum(n_rejected_list) / total_spec_events) if total_spec_events else 0

    return {
        "n": len(instances),
        "avg_score": statistics.mean(scores) if scores else 0,
        "success_rate": (sum(1 for s in scores if s > 0) / len(scores)) if scores else 0,
        "total_spec_accepted": sum(n_accepted_list),
        "total_spec_rejected": sum(n_rejected_list),
        "total_summary_fallback": sum(n_fallback_list),
        "reject_rate": reject_rate,
        "summary_ratio": _calc_stats(summary_ratios),
        "judge_ratio": _calc_stats(judge_ratios),
        "num_steps": _calc_stats(num_steps_list),
        "speculative_steps": _calc_stats(speculative_steps_list),
        "total_time": _calc_stats(total_times),
        "total_tokens": _calc_stats(total_tokens),
        "agent_tokens": _calc_stats(agent_tokens_list),
        "agent_time": _calc_stats(agent_times_list),
        "action_tokens": _calc_stats(action_tokens_list),
        "action_time": _calc_stats(action_times_list),
        "summary_tokens": _calc_stats(summary_tokens_list),
        "summary_time": _calc_stats(summary_times_list),
        "judge_time": _calc_stats(judge_times_list),
        "speculation_overlap": _calc_stats(speculation_overlaps),
        "spec_sum_time": _calc_stats(spec_sum_times),
        "spec_steps_time": _calc_stats(spec_steps_times),
    }


def _write_stats_block(f, stats: dict) -> None:
    metrics = [
        ("Summary Ratio", stats["summary_ratio"]),
        ("Num of Steps", stats["num_steps"]),
        ("Speculative Steps", stats["speculative_steps"]),
        ("Total Time (s)", stats["total_time"]),
        ("Total Tokens", stats["total_tokens"]),
        ("Agent Tokens", stats["agent_tokens"]),
        ("Agent Time (s)", stats["agent_time"]),
        ("Action Tokens", stats["action_tokens"]),
        ("Action Time (s)", stats["action_time"]),
        ("Summary Tokens", stats["summary_tokens"]),
        ("Summary Time (s)", stats["summary_time"]),
        ("Judge Ratio", stats["judge_ratio"]),
        ("Judge Time (s)", stats["judge_time"]),
        ("Spec Summary Time (s)", stats["spec_sum_time"]),
        ("Spec Steps Time (s)", stats["spec_steps_time"]),
    ]

    f.write(f"N={stats['n']}  Avg score: {stats['avg_score']:.4f}  Success rate: {stats['success_rate']:.4f}\n")
    f.write(
        f"Spec accepted: {stats['total_spec_accepted']}  rejected: {stats['total_spec_rejected']}  "
        f"fallback: {stats['total_summary_fallback']}  reject rate: {stats['reject_rate']:.4f}\n\n"
    )
    f.write(f"{'Metric':<22} {'Mean':<15} {'P50':<15}\n")
    f.write("-" * 52 + "\n")

    for metric_name, s in metrics:
        mean_val = s["mean"]
        p50_val = s["p50"]
        if "ratio" in metric_name.lower() or "time" in metric_name.lower():
            f.write(f"{metric_name:<22} {mean_val:<15.4f} {p50_val:<15.4f}\n")
        else:
            f.write(f"{metric_name:<22} {mean_val:<15.2f} {p50_val:<15.2f}\n")

    seq = (
        stats["agent_time"]["mean"]
        + stats["action_time"]["mean"]
        + stats["summary_time"]["mean"]
        + stats["judge_time"]["mean"]
    )
    savings = stats["spec_steps_time"]["mean"]
    f.write("\nConsistency check:\n")
    f.write(f"  Agent + Action + Summary + Judge = {seq:.2f}s\n")
    f.write(f"  - Spec Steps Time (savings)      = {savings:.2f}s\n")
    f.write(f"  = Predicted Total Time           = {seq - savings:.2f}s\n")
    f.write(f"  Actual Total Time (mean)         = {stats['total_time']['mean']:.2f}s\n")


def write_overall_stats(instances: list[dict], total_program_time: float, output_file: Path) -> None:
    all_stats = _collect_stats(instances)
    async_instances = [inst for inst in instances if inst["total_speculative_steps"] > 0]
    async_stats = _collect_stats(async_instances) if async_instances else None

    with output_file.open("w", encoding="utf-8") as f:
        f.write("Overall Statistics\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"Longest instance wall time: {total_program_time:.2f}s, {total_program_time / 60:.2f} minutes\n\n")
        _write_stats_block(f, all_stats)

        if async_stats:
            f.write("\n\nAsync Summary Instances (speculative_steps > 0)\n")
            f.write("=" * 52 + "\n\n")
            _write_stats_block(f, async_stats)

    print(f"Created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path, help="Directory containing per-instance <inst>/<inst>.stats.json files")
    parser.add_argument("--eval-tag", type=str, default=None, help="Subdir name under eval_logs/ to pull report.json scores from (default: prefer 'final', else the newest)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Where to write the TSV + overall_stats.txt (default: <results_dir>/summary/)")
    args = parser.parse_args()

    results_dir: Path = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    eval_tag = _pick_eval_tag(results_dir, args.eval_tag)
    if eval_tag is None:
        print(f"[warn] no eval_logs/ under {results_dir} — all scores will be 0")
    else:
        print(f"Using eval_logs/{eval_tag}/ for scores")

    print(f"Parsing {results_dir}...")
    instances, total_program_time = parse_results_dir(results_dir, eval_tag)
    if not instances:
        raise SystemExit(f"No *.stats.json files found under {results_dir}")
    print(f"Found {len(instances)} instances")

    output_dir = (args.output_dir or (results_dir / "summary")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / f"{results_dir.name}.tsv"
    create_csv_files(instances, summary_file)

    overall_stats_file = output_dir / "overall_stats.txt"
    write_overall_stats(instances, total_program_time, overall_stats_file)

    print("\nDone! Files created:")
    print(f"  - {summary_file}")
    print(f"  - {overall_stats_file}")


if __name__ == "__main__":
    main()
