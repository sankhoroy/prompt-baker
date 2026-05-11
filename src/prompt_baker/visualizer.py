from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def plot_progress(run_dir: str | Path, output_file: str | Path | None = None) -> Path:
    run_path = Path(run_dir)
    score_file = _resolve_scores_file(run_path)
    create_scores_csv(score_file.parent)

    points: list[tuple[int, float]] = []
    with score_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            generation = int(record["generation"])
            score = float(record["candidate"]["score"] or 0.0)
            points.append((generation, score))

    if not points:
        raise ValueError("No score points found in score log.")

    best_so_far: list[float] = []
    current_best = 0.0
    for _, score in points:
        current_best = max(current_best, score)
        best_so_far.append(current_best)

    x_axis = list(range(1, len(points) + 1))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(x_axis, [score for _, score in points], label="score", alpha=0.6)
    axis.plot(x_axis, best_so_far, label="best_so_far", linewidth=2.0)
    axis.set_title("Prompt Baker Optimization Progress")
    axis.set_xlabel("Candidate Evaluation Order")
    axis.set_ylabel("Score")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.3)

    out_path = Path(output_file) if output_file else run_path / "progress.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def create_scores_csv(run_dir: str | Path, output_file: str | Path | None = None) -> Path:
    run_path = Path(run_dir)
    score_file = _resolve_scores_file(run_path)

    rows: list[dict[str, str | int | float]] = []
    with score_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            candidate = record.get("candidate", {})
            metrics = candidate.get("metrics", {})
            metadata = candidate.get("metadata", {})
            rows.append(
                {
                    "timestamp": str(record.get("timestamp", "")),
                    "generation": int(record.get("generation", 0)),
                    "candidate_idx": int(record.get("candidate_idx", 0)),
                    "score": float(candidate.get("score") or 0.0),
                    "base_score": float(metrics.get("base_score") or 0.0),
                    "final_score": float(metrics.get("final_score") or 0.0),
                    "model_name": str(candidate.get("model_name", "")),
                    "system_prompt": str(candidate.get("system_prompt", "")),
                    "user_prompt": str(candidate.get("user_prompt", "")),
                    "system_original_prompt": str(metadata.get("system_original_prompt", "")),
                    "system_paraphrased_prompt": str(metadata.get("system_paraphrased_prompt", "")),
                    "user_original_prompt": str(metadata.get("user_original_prompt", "")),
                    "user_paraphrased_prompt": str(metadata.get("user_paraphrased_prompt", "")),
                }
            )

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["score"],
            _timestamp_to_epoch(str(row["timestamp"])),
        ),
        reverse=True,
    )

    out_path = Path(output_file) if output_file else score_file.parent / "scores_sorted.csv"
    fieldnames = [
        "timestamp",
        "generation",
        "candidate_idx",
        "score",
        "base_score",
        "final_score",
        "model_name",
        "system_prompt",
        "user_prompt",
        "system_original_prompt",
        "system_paraphrased_prompt",
        "user_original_prompt",
        "user_paraphrased_prompt",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)
    return out_path


def _resolve_scores_file(run_path: Path) -> Path:
    direct_scores = run_path / "scores.jsonl"
    if direct_scores.exists():
        return direct_scores

    run_candidates = sorted(
        (
            item
            for item in run_path.glob("run_*")
            if item.is_dir() and (item / "scores.jsonl").exists()
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if run_candidates:
        return run_candidates[0] / "scores.jsonl"

    raise FileNotFoundError(
        f"Could not find scores log in '{run_path}'. "
        "Pass a run directory containing scores.jsonl or a logs directory containing run_* folders."
    )


def _timestamp_to_epoch(timestamp: str) -> float:
    if not timestamp:
        return float("-inf")
    try:
        return datetime.fromisoformat(timestamp).timestamp()
    except ValueError:
        return float("-inf")
