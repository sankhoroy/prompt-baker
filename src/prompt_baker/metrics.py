from __future__ import annotations

from statistics import mean
from typing import Sequence

from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from prompt_baker.types import JudgeScoreFn


def classification_metric(y_true: Sequence[str], y_pred: Sequence[str], metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_score":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(f"Unsupported classification metric: {metric}. please implement custom metric and plug it in.")


def rouge_metric(y_true: Sequence[str], y_pred: Sequence[str], metric: str) -> float:
    mapping = {"rouge-1": "rouge1", "rouge-2": "rouge2", "rouge-l": "rougeL"}
    scorer = rouge_scorer.RougeScorer([mapping[metric]], use_stemmer=True)
    scores = [scorer.score(reference, candidate)[mapping[metric]].fmeasure for reference, candidate in zip(y_true, y_pred)]
    return float(mean(scores)) if scores else 0.0


def embedding_similarity_metric(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError(
            "embedding_similarity requires sentence-transformers. "
            "Install it with: uv add sentence-transformers"
        ) from exc
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    true_emb = model.encode(list(y_true))
    pred_emb = model.encode(list(y_pred))
    sim = cosine_similarity(true_emb, pred_emb)
    diagonal_scores = [float(sim[idx][idx]) for idx in range(len(y_true))]
    return float(mean(diagonal_scores)) if diagonal_scores else 0.0


def llm_judge_metric(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    judge_score_fn: JudgeScoreFn,
    scale_max: float = 10.0,
) -> float:
    all_scores: list[float] = []
    for reference, candidate in zip(y_true, y_pred):
        score = judge_score_fn(reference, candidate, scale_max)
        all_scores.append(score)
    return float(mean(all_scores)) if all_scores else 0.0
