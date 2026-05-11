from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm

from prompt_baker.logging import LogTracker
from prompt_baker.metrics import (
    classification_metric,
    embedding_similarity_metric,
    llm_judge_metric,
    rouge_metric,
)
from prompt_baker.types import (
    CandidateGene,
    ChatModelSpec,
    JudgeScoreFn,
    OptimizerConfig,
    ParaphraseFn,
)


class PromptBakerOptimizer:
    def __init__(
        self,
        model_specs: list[ChatModelSpec],
        system_prompts: list[str],
        user_prompts: list[str],
        config: OptimizerConfig,
        paraphrase_fn: ParaphraseFn | None = None,
        judge_score_fn: JudgeScoreFn | None = None,
        paraphrase_completion_fn: Callable[[str], str] | None = None,
        paraphrase_cache_file: str | Path = "paraphrase_cache.json",
        run_dir: str | Path = "logs",
    ) -> None:
        if not model_specs:
            raise ValueError("model_specs cannot be empty.")

        if not system_prompts or not user_prompts:
            raise ValueError("system_prompts and user_prompts cannot be empty.")

        self.model_specs = {spec.name: spec for spec in model_specs}
        self.system_prompts = system_prompts
        self.user_prompts = user_prompts
        self.config = config

        self.paraphrase_completion_fn = paraphrase_completion_fn
        self.paraphrase_fn = paraphrase_fn or self.simple_paraphrase

        self.judge_score_fn = judge_score_fn

        self._rng = random.Random(config.random_seed)

        self._prompt_cache: dict[tuple[str, bool, str], list[str]] = {}

        self._system_prompt_meta: dict[str, dict[str, str]] = {}
        self._user_prompt_meta: dict[str, dict[str, str]] = {}

        self.logger = LogTracker(run_dir)

        # =====================================================
        # PARAPHRASE CACHE
        # =====================================================

        self.paraphrase_cache_file = Path(paraphrase_cache_file)

        if self.paraphrase_cache_file.exists():
            with open(self.paraphrase_cache_file, "r", encoding="utf-8") as f:
                self._persistent_paraphrase_cache = json.load(f)
        else:
            self._persistent_paraphrase_cache = {}

    def _save_paraphrase_cache(self) -> None:
        with open(self.paraphrase_cache_file, "w", encoding="utf-8") as f:
            json.dump(
                self._persistent_paraphrase_cache,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def optimize(self, dataset_csv: str | Path, verbose: bool = False) -> CandidateGene:
        dataset = pd.read_csv(dataset_csv)
        population = self._initial_population()

        print( f"Baking process started ... 📄📄📄📄📄 ➜ 🥣 ➜ ♨️🔥🔥🔥\n")
        self.logger.log_event("run_started", {"rows": len(dataset), "config": asdict(self.config)})

        best_candidate: CandidateGene | None = None
        generation_iter = tqdm(
            range(self.config.generations),
            desc="Generations",
            disable=not verbose,
            leave=True,
        )

        for generation in generation_iter:
            evaluated = self._evaluate_population(
                population=population,
                dataset=dataset,
                generation=generation,
                verbose=verbose,
            )

            evaluated.sort(key=lambda gene: gene.score or 0.0, reverse=True)

            if not best_candidate or (evaluated[0].score or 0.0) > (best_candidate.score or 0.0):
                best_candidate = evaluated[0]

            if verbose:
                current_best = evaluated[0]

                generation_iter.set_postfix(
                    {
                        "metric": self.config.metric,
                        "best_score": f"{(current_best.score or 0.0):.4f}",
                        "best_model": current_best.model_name,
                        "best_system": self._truncate_text(current_best.system_prompt, max_len=28),
                        "best_user": self._truncate_text(current_best.user_prompt, max_len=28),
                    }
                )

            self.logger.log_event(
                "generation_completed",
                {
                    "generation": generation,
                    "best_score": evaluated[0].score,
                    "best_metrics": evaluated[0].metrics,
                },
            )

            if generation < self.config.generations - 1:
                population = self._next_generation(evaluated)

        print("🍞 Prompt model combination baked(optimized) successfully ✨\n")

        assert best_candidate is not None

        self.logger.write_summary(
            {
                "best_candidate": asdict(best_candidate),
                "run_dir": str(self.logger.run_dir),
                "rows": len(dataset),
            }
        )

        return best_candidate

    def _initial_population(self) -> list[CandidateGene]:
        all_system = self._expand_prompts(self.system_prompts, prompt_kind="system")
        all_user = self._expand_prompts(self.user_prompts, prompt_kind="user")

        model_names = list(self.model_specs.keys())

        population: list[CandidateGene] = []

        for _ in range(self.config.population_size):
            system_prompt = self._rng.choice(all_system)
            user_prompt = self._rng.choice(all_user)

            population.append(
                CandidateGene(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=self._rng.choice(model_names),
                    metadata=self._build_candidate_metadata(system_prompt, user_prompt),
                )
            )

        return population

    def _expand_prompts(self, prompts: list[str], prompt_kind: str) -> list[str]:
        expanded: list[str] = []

        target_meta = (
            self._system_prompt_meta
            if prompt_kind == "system"
            else self._user_prompt_meta
        )

        for prompt in prompts:
            original = prompt.strip()

            if not original:
                continue

            expanded.append(original)

            target_meta.setdefault(
                original,
                {
                    "original_prompt": original,
                    "paraphrased_prompt": original,
                    "prompt_kind": prompt_kind,
                    "variant_type": "original",
                },
            )

            expanded.extend(
                self._paraphrase(
                    original,
                    concise=False,
                    prompt_kind=prompt_kind,
                )
            )

            if self.config.token_length_optimisation:
                expanded.extend(
                    self._paraphrase(
                        original,
                        concise=True,
                        prompt_kind=prompt_kind,
                    )
                )

        deduped = list(
            dict.fromkeys(
                item.strip()
                for item in expanded
                if item.strip()
            )
        )

        return deduped

    def _paraphrase(
        self,
        prompt: str,
        concise: bool,
        prompt_kind: str,
    ) -> list[str]:

        key = (prompt, concise, prompt_kind)

        if key in self._prompt_cache:
            return self._prompt_cache[key]

        if self.config.paraphrases_per_prompt <= 0:
            self._prompt_cache[key] = []
            return []

        results: list[str] = []

        target_meta = (
            self._system_prompt_meta
            if prompt_kind == "system"
            else self._user_prompt_meta
        )

        for _ in range(self.config.paraphrases_per_prompt):
            text = self.paraphrase_fn(prompt, concise).strip()

            text = self._with_original_prompt(text, prompt)

            if text:
                results.append(text)

                target_meta.setdefault(
                    text,
                    {
                        "original_prompt": prompt,
                        "paraphrased_prompt": text,
                        "prompt_kind": prompt_kind,
                        "variant_type": (
                            "paraphrased_concise"
                            if concise
                            else "paraphrased"
                        ),
                    },
                )

        self._prompt_cache[key] = results

        return results

    def simple_paraphrase(self, prompt: str, concise: bool) -> str:
        mode = "concise" if concise else "normal"

        cache_key = f"{mode}::{prompt}"

        # =====================================================
        # CACHE HIT
        # =====================================================

        if cache_key in self._persistent_paraphrase_cache:
            cached_variants = self._persistent_paraphrase_cache[cache_key]

            if cached_variants:
                return self._rng.choice(cached_variants)

        # =====================================================
        # STRATEGY
        # =====================================================

        paraphrase_strategies = [
            "Rephrase this task while preserving exact meaning, constraints, and output format.",
            "Rewrite this prompt for clarity while keeping the task identical.",
            "Break down this task using chain-of-thought reasoning.",
            "Rewrite this using tree-of-thought decomposition.",
            "Improve this prompt for an AI agent while preserving behavior.",
            "Make the task instructions more explicit and structured.",
        ]

        concise_strategies = [
            "Make this prompt as short as possible without losing meaning or output format.",
            "Compress this task aggressively while preserving all requirements.",
            "Reduce token count while preserving exact behavior.",
        ]

        instruction = (
            self._rng.choice(concise_strategies)
            if concise
            else self._rng.choice(paraphrase_strategies)
        )

        full_prompt = f"""
You are an expert prompt engineer.

TASK:
{instruction}

RULES:
- Preserve intent
- Preserve constraints
- Preserve output format
- Preserve task meaning
- Return ONLY rewritten prompt

ORIGINAL PROMPT:
{prompt}

REWRITTEN PROMPT:
""".strip()

        # =====================================================
        # LLM CALL
        # =====================================================

        if self.paraphrase_completion_fn is None:
            raise ValueError(
                "paraphrase_completion_fn must be provided."
            )

        rewritten = self.paraphrase_completion_fn(full_prompt).strip()

        # =====================================================
        # SAVE CACHE
        # =====================================================

        if rewritten:
            self._persistent_paraphrase_cache.setdefault(
                cache_key,
                []
            )

            if rewritten not in self._persistent_paraphrase_cache[cache_key]:
                self._persistent_paraphrase_cache[cache_key].append(
                    rewritten
                )

                self._save_paraphrase_cache()

            return rewritten

        return prompt

    @staticmethod
    def _with_original_prompt(rephrased: str, original: str) -> str:
        rephrased_clean = rephrased.strip()
        original_clean = original.strip()

        if not rephrased_clean:
            return ""

        if original_clean.lower() in rephrased_clean.lower():
            return rephrased_clean

        return f"{rephrased_clean}\n\nOriginal prompt:\n{original_clean}"

    def _evaluate_population(
        self,
        population: list[CandidateGene],
        dataset: pd.DataFrame,
        generation: int,
        verbose: bool = False,
    ) -> list[CandidateGene]:

        evaluated: list[CandidateGene] = []

        best_in_generation: CandidateGene | None = None

        if verbose:
            candidate_bar = tqdm(
                enumerate(population),
                total=len(population),
                desc=f"Generation {generation + 1} candidates",
                disable=not verbose,
                leave=False,
            )

            candidate_iter = candidate_bar

        else:
            candidate_bar = None
            candidate_iter = enumerate(population)

        for idx, candidate in candidate_iter:
            if candidate_bar is not None:
                candidate_bar.set_postfix(
                    {
                        "running_model": candidate.model_name,
                        "running_system": self._truncate_text(candidate.system_prompt, max_len=24),
                        "running_user": self._truncate_text(candidate.user_prompt, max_len=24),
                    }
                )

            y_true, y_pred = self._predict_dataset(candidate, dataset)

            score, metrics = self._score_candidate(candidate, y_true, y_pred)

            candidate.score = score
            candidate.metrics = metrics

            self.logger.log_candidate(
                generation=generation,
                candidate_idx=idx,
                candidate=candidate,
            )

            evaluated.append(candidate)

            if not best_in_generation or (candidate.score or 0.0) > (best_in_generation.score or 0.0):
                best_in_generation = candidate

            if candidate_bar is not None and best_in_generation is not None:
                candidate_bar.set_postfix(
                    {
                        "running_model": candidate.model_name,
                        "running_system": self._truncate_text(candidate.system_prompt, max_len=18),
                        "running_user": self._truncate_text(candidate.user_prompt, max_len=18),
                        "best_model": best_in_generation.model_name,
                        "best_score": f"{(best_in_generation.score or 0.0):.4f}",
                    }
                )

        return evaluated

    def _predict_dataset(
        self,
        candidate: CandidateGene,
        dataset: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:

        completion_fn = self.model_specs[candidate.model_name].completion_fn

        y_true: list[str] = []
        y_pred: list[str] = []

        for _, row in dataset.iterrows():
            row_input = str(row[self.config.input_column])
            row_target = str(row[self.config.target_column])

            user_prompt = self._render_user_prompt(
                candidate.user_prompt,
                row_input,
            )

            predicted = completion_fn(
                candidate.system_prompt,
                user_prompt,
            ).strip()

            y_true.append(row_target)
            y_pred.append(predicted)

        return y_true, y_pred

    def _render_user_prompt(self, template: str, row_input: str) -> str:
        if "{input}" in template:
            return template.format(input=row_input)

        return f"{template}\n\nInput:\n{row_input}"

    def _score_candidate(
        self,
        candidate: CandidateGene,
        y_true: list[str],
        y_pred: list[str],
    ) -> tuple[float, dict[str, float]]:

        metric_name = self.config.metric

        if self.config.task_type == "classification":
            y_true_norm = [item.strip().lower() for item in y_true]
            y_pred_norm = [item.strip().lower() for item in y_pred]

            base_score = classification_metric(
                y_true_norm,
                y_pred_norm,
                metric_name,
            )

        elif metric_name in {"rouge-1", "rouge-2", "rouge-l"}:
            base_score = rouge_metric(y_true, y_pred, metric_name)

        elif metric_name == "embedding_similarity":
            base_score = embedding_similarity_metric(y_true, y_pred)

        elif metric_name == "llm_as_judge":
            if not self.judge_score_fn:
                raise ValueError(
                    "judge_score_fn is required for llm_as_judge metric."
                )

            base_score = llm_judge_metric(
                y_true,
                y_pred,
                self.judge_score_fn,
                scale_max=self.config.judge_scale_max,
            )

        else:
            raise ValueError(
                f"Unsupported metric {metric_name} for task type {self.config.task_type}"
            )

        score = base_score

        if self.config.token_length_optimisation:
            prompt_len = (
                len(candidate.system_prompt)
                + len(candidate.user_prompt)
            )

            penalty = (
                min(0.5, prompt_len / 4000.0)
                * self.config.token_penalty_weight
            )

            score = max(0.0, base_score - penalty)

        metrics = {
            "base_score": float(base_score),
            "final_score": float(score),
        }

        return float(score), metrics

    def _next_generation(
        self,
        ranked_population: list[CandidateGene],
    ) -> list[CandidateGene]:

        elite = ranked_population[: self.config.elite_size]

        next_population: list[CandidateGene] = [
            self._clone_gene(item)
            for item in elite
        ]

        while len(next_population) < self.config.population_size:
            parent1, parent2 = self._select_parents(ranked_population)

            child = (
                self._crossover(parent1, parent2)
                if self._rng.random() < self.config.crossover_rate
                else self._clone_gene(parent1)
            )

            if self._rng.random() < self.config.mutation_rate:
                child = self._mutate(child)

            next_population.append(child)

        return next_population[: self.config.population_size]

    def _select_parents(
        self,
        population: list[CandidateGene],
    ) -> tuple[CandidateGene, CandidateGene]:

        tournament_size = min(4, len(population))

        choices = self._rng.sample(population, tournament_size)

        choices.sort(
            key=lambda gene: gene.score or 0.0,
            reverse=True,
        )

        return choices[0], choices[1 if len(choices) > 1 else 0]

    def _crossover(
        self,
        parent1: CandidateGene,
        parent2: CandidateGene,
    ) -> CandidateGene:

        system_prompt = (
            parent1.system_prompt
            if self._rng.random() < 0.5
            else parent2.system_prompt
        )

        user_prompt = (
            parent1.user_prompt
            if self._rng.random() < 0.5
            else parent2.user_prompt
        )

        model_name = (
            parent1.model_name
            if self._rng.random() < 0.5
            else parent2.model_name
        )

        return CandidateGene(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=model_name,
            metadata=self._build_candidate_metadata(
                system_prompt,
                user_prompt,
            ),
        )

    def _mutate(self, gene: CandidateGene) -> CandidateGene:
        system_pool = self._expand_prompts(
            self.system_prompts,
            prompt_kind="system",
        )

        user_pool = self._expand_prompts(
            self.user_prompts,
            prompt_kind="user",
        )

        model_pool = list(self.model_specs.keys())

        mutation_ops: list[Callable[[CandidateGene], None]] = [
            lambda g: setattr(g, "system_prompt", self._rng.choice(system_pool)),
            lambda g: setattr(g, "user_prompt", self._rng.choice(user_pool)),
            lambda g: setattr(g, "model_name", self._rng.choice(model_pool)),
        ]

        mutated = self._clone_gene(gene)

        self._rng.choice(mutation_ops)(mutated)

        mutated.metadata = self._build_candidate_metadata(
            mutated.system_prompt,
            mutated.user_prompt,
        )

        return mutated

    @staticmethod
    def _truncate_text(text: str, max_len: int = 120) -> str:
        if len(text) <= max_len:
            return text

        return text[: max_len - 3] + "..."

    @staticmethod
    def _clone_gene(gene: CandidateGene) -> CandidateGene:
        return CandidateGene(
            system_prompt=gene.system_prompt,
            user_prompt=gene.user_prompt,
            model_name=gene.model_name,
            score=gene.score,
            metrics=dict(gene.metrics),
            metadata=dict(gene.metadata),
        )

    def _build_candidate_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, str]:

        system_meta = self._system_prompt_meta.get(
            system_prompt,
            {
                "original_prompt": system_prompt,
                "paraphrased_prompt": system_prompt,
                "prompt_kind": "system",
                "variant_type": "unknown",
            },
        )

        user_meta = self._user_prompt_meta.get(
            user_prompt,
            {
                "original_prompt": user_prompt,
                "paraphrased_prompt": user_prompt,
                "prompt_kind": "user",
                "variant_type": "unknown",
            },
        )

        return {
            "system_original_prompt": system_meta["original_prompt"],
            "system_paraphrased_prompt": system_meta["paraphrased_prompt"],
            "system_variant_type": system_meta["variant_type"],
            "user_original_prompt": user_meta["original_prompt"],
            "user_paraphrased_prompt": user_meta["paraphrased_prompt"],
            "user_variant_type": user_meta["variant_type"],
        }