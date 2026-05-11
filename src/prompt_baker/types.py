from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

TaskType = Literal["classification", "generation"]
MetricName = Literal[
    "accuracy",
    "f1_score",
    "precision",
    "recall",
    "rouge-1",
    "rouge-2",
    "rouge-l",
    "embedding_similarity",
    "llm_as_judge",
]

CompletionFn = Callable[[str, str], str]
ParaphraseFn = Callable[[str, bool], str]
JudgeScoreFn = Callable[[str, str, float], float]


@dataclass(slots=True)
class ChatModelSpec:
    name: str
    completion_fn: CompletionFn


@dataclass(slots=True)
class CandidateGene:
    system_prompt: str
    user_prompt: str
    model_name: str
    score: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizerConfig:
    task_type: TaskType
    metric: MetricName
    population_size: int = 12
    generations: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.6
    elite_size: int = 2
    random_seed: int = 42
    input_column: str = "input"
    target_column: str = "target"
    token_length_optimisation: bool = False
    token_penalty_weight: float = 0.03
    paraphrases_per_prompt: int = 2
    judge_scale_max: float = 10.0
