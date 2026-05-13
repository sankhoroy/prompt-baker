<!-- <img src="logo.png" style="max-width:100%; height:auto;" width="200"> -->
<!-- ![prompt-baker](assets/prompt-baker-logo-2.png) -->
<img src="assets/prompt-baker-logo.png" alt="prompt-baker" width="220" height="220">

# prompt-baker

**Prompt-Baker** helps you automatically discover the best prompt, model, and parameter combinations for your LLM workflows instead of manually testing endless variations. Just provide your task, evaluation dataset (like a CSV benchmark), and candidate prompts/models — Prompt-Baker experiments across combinations to find what actually performs best for your use case.

Whether you're building RAG pipelines, AI agents, tool-calling systems, classifiers, or generation apps, Prompt-Baker can evolve better prompts, optimize retriever strategies, tune inference settings, and improve agent behavior over time. It works with any chat completion function — APIs, local models, LangChain agents, heuristics, or custom pipelines — making it easy to optimize real-world AI systems automatically.

## Demo
**Get best system_prompt, user_prompt and model cobmination to get `Sentiment Classification` from a golden(benchmark) dataset**: https://colab.research.google.com/drive/1NGbnGoMNiGiCqN9V_WhuFY64cdh2KZJ6?usp=sharing

**Get best system prompt, user prompt, retriver k of RAG, llm model from combinations for `Cat–dog RAG`** : https://colab.research.google.com/drive/1CERUoTUw972n9S5e23lkUNh9rhvIQ55H?usp=sharing

## Tutorials
Video tutorials: https://www.youtube.com/@prompt-baker


## Features

- Pluggable completion backends via `ChatModelSpec`
- Pools of system prompts, user prompts (with `{input}`), and multiple models in one run
- Tasks: **classification** and **generation**
- Metrics: accuracy, F1, precision, recall; ROUGE; optional embedding similarity and LLM-as-judge
- JSONL logging per run and helpers to plot progress and export score tables

## Requirements

- Python 3.10 or newer

## Installation

### From PyPI (pip)

```bash
pip install prompt-baker
```

### From PyPI (uv)

```bash
uv add prompt-baker
```

### From source (GitHub clone)

```bash
git clone https://github.com/YOUR_USERNAME/prompt-baker.git
cd prompt-baker
```

Then either:

**uv (recommended for this repo)**

```bash
uv sync
uv run prompt-baker --about
```

**pip in a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
prompt-baker --about
```

Replace `YOUR_USERNAME` in the clone URL with your GitHub user or organization. Before publishing, update `authors` and `project.urls` in `pyproject.toml` to match your PyPI and GitHub accounts.

## Repository layout

```
prompt-baker/
├── pyproject.toml          # Package metadata, build (hatchling), uv dev deps
├── LICENSE
├── README.md
├── src/
│   └── prompt_baker/
│       ├── __init__.py     # PromptBakerOptimizer, ChatModelSpec, OptimizerConfig
│       ├── optimizer.py
│       ├── types.py
│       ├── metrics.py
│       ├── logging.py
│       ├── visualizer.py
│       └── cli.py          # prompt-baker console entry point
├── scripts/
│   └── visualize_logs.py   # CLI to plot a run and emit scores CSV
├── tests/
└── examples/               # See examples/README.md
    ├── sentiment/
    └── rag_cat_dog/
```

## Command-line tools

After installation, the package exposes:

```bash
prompt-baker --about
```

Optimization is driven from Python. To visualize an existing run directory (contains `scores.jsonl`):

```bash
python scripts/visualize_logs.py --run-dir logs/run_YYYYMMDD_HHMMSS
```

If you installed with `uv` from the repo root, you can use `uv run python scripts/visualize_logs.py ...`.

## Python API (minimal example)

```python
from prompt_baker import ChatModelSpec, OptimizerConfig, PromptBakerOptimizer


def my_completion(system_prompt: str, user_prompt: str) -> str:
    # Inject any backend: HTTP API, LangChain agent, local model, etc.
    return "positive"


models = [
    ChatModelSpec(
        name="my-backend",
        completion_fn=my_completion,
    )
]

config = OptimizerConfig(
    task_type="classification",
    metric="accuracy",
    generations=4,
    population_size=10,
    token_length_optimisation=True,
)

optimizer = PromptBakerOptimizer(
    model_specs=models,
    system_prompts=[
        "You are a strict classifier. Return only one label.",
    ],
    user_prompts=[
        "Classify this text into {input}",
        "Given input: {input}\nReturn only class label.",
    ],
    config=config,
    paraphrase_fn=lambda prompt, concise: f"Briefly: {prompt}" if concise else f"Rewrite: {prompt}",
)

best = optimizer.optimize("data/golden.csv")
print(best)
```

### LangChain-style agent adapter

```python
def build_completion_from_agent(agent):
    def completion(system_prompt: str, user_prompt: str) -> str:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            }
        )
        return str(result["messages"][-1].content).strip()

    return completion
```

### Dataset contract

CSV columns (defaults can be overridden on `OptimizerConfig`):

- **input** — text fed into `{input}` in user prompts (or `input_column`)
- **target** — gold label or reference string (or `target_column`)

## Metrics

**Classification:** `accuracy`, `f1_score`, `precision`, `recall`

**Generation:** `rouge-1`, `rouge-2`, `rouge-l`; optional `embedding_similarity` (needs `sentence-transformers`); `llm_as_judge` (requires a `judge_score_fn` on the optimizer)

## Logs and visualization

Each optimization run writes a directory such as `logs/run_YYYYMMDD_HHMMSS` with:

- `events.jsonl`
- `scores.jsonl`
- `summary.json`

From the repo, generate a plot and CSV in a separate process:

```bash
uv run python scripts/visualize_logs.py --run-dir logs/run_YYYYMMDD_HHMMSS
```

Or call `plot_progress` / `create_scores_csv` from `prompt_baker.visualizer` in code (see the examples).

## Examples

Two worked examples live under `examples/`:

| Example | Directory | Notebook | Summary |
|--------|-----------|----------|---------|
| Sentiment classification | `examples/sentiment/` | [sentiment_classification.ipynb](https://github.com/sankhoroy/prompt-baker/blob/main/examples/sentiment/sentiment_classification.ipynb) | CSV benchmark, genetic search over prompts and backends, optional Groq agent |
| Cat–dog RAG | `examples/rag_cat_dog/` | [cat_dog_rag.ipynb](https://github.com/sankhoroy/prompt-baker/blob/main/examples/rag_cat_dog/cat_dog_rag.ipynb) | Chroma retrieval, multiple retriever strategies, LLM-as-judge |

See **[examples/README.md](examples/README.md)** for paths, extra dependencies, and how to run each script or notebook.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

## Contact

For questions, bug reports, or feature ideas, email [sankhoroy@gmail.com](mailto:sankhoroy@gmail.com).

## License

MIT — see [LICENSE](LICENSE).
