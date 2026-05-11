from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prompt_baker.types import CandidateGene


class LogTracker:
    def __init__(self, output_dir: str | Path = "logs") -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.run_dir / "events.jsonl"
        self.scores_file = self.run_dir / "scores.jsonl"
        self.summary_file = self.run_dir / "summary.json"

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def log_candidate(self, generation: int, candidate_idx: int, candidate: CandidateGene) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "generation": generation,
            "candidate_idx": candidate_idx,
            "candidate": asdict(candidate),
        }
        with self.scores_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        with self.summary_file.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=True)
