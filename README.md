<p align="center">
  <img src="assets/prompt-baker-logo.png" alt="prompt-baker" width="480">
</p>

# prompt-baker

Genetic optimization for **prompt + model** combinations on CSV benchmarks. You inject chat completion functions (any API, local model, LangChain agent, or heuristic), supply pools of system and user prompt templates, and search for high-scoring candidates using classification or generation metrics.

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

| Example | Directory | Summary |
|--------|-----------|---------|
| Sentiment classification | `examples/sentiment/` | CSV benchmark, genetic search over prompts and backends, optional Groq agent |
| Cat–dog RAG | `examples/rag_cat_dog/` | Chroma retrieval, multiple retriever strategies, LLM-as-judge |

See **[examples/README.md](examples/README.md)** for paths, extra dependencies, and how to run each script or notebook.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

## Publishing (GitHub, PyPI, pip, uv)

### 1. GitHub

1. Create a new empty repository on GitHub (no README if you already have one locally).
2. Point your local tree at it and push (use your real URL):

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/prompt-baker.git
   git branch -M main
   git add .
   git commit -m "Initial release"
   git push -u origin main
   ```

3. In `pyproject.toml`, set `[project.urls]` `Homepage`, `Repository`, and `Issues` to that repo, and set `authors` to your name and email.

### 2. PyPI (one-time)

1. Create accounts on [pypi.org](https://pypi.org/account/register/) and (optional) [test.pypi.org](https://test.pypi.org/account/register/).
2. Enable **2FA** on PyPI (required to upload).
3. Create an **API token** (scope: whole account or project `prompt-baker` once the name is reserved). Treat the token like a password.

Optional: use [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) from GitHub Actions so you never store a long-lived token locally.

### 3. Build the distribution

From the repository root (with dev deps so `build` / `twine` are available):

```bash
uv sync
uv run python -m build
```

This writes `dist/prompt_baker-0.1.0-py3-none-any.whl` and `dist/prompt_baker-0.1.0.tar.gz` (names follow the current `version` in `pyproject.toml`).

Check the artifacts:

```bash
uv run twine check dist/*
```

### 4. Upload to PyPI

**Option A — Twine (classic)**

```bash
uv run twine upload dist/*
```

When prompted, PyPI username is `__token__` and the password is your API token (including the `pypi-` prefix). You can also set `TWINE_USERNAME` and `TWINE_PASSWORD` in the environment.

**Option B — uv**

```bash
uv publish
```

`uv` will use PyPI credentials from its [keyring / token configuration](https://docs.astral.sh/uv/guides/package/#publishing-your-package). First-time setup is described in that guide.

Test uploads can target TestPyPI first:

```bash
uv run twine upload --repository testpypi dist/*
```

### 5. After it is on PyPI

Anyone can install the same artifact with **pip** or **uv**:

```bash
pip install prompt-baker
```

```bash
uv add prompt-baker
```

Bump `version` in `pyproject.toml` (and `src/prompt_baker/__init__.py` if you keep them in sync) before each new release, rebuild `dist/`, and upload again. Delete old `dist/` between releases if you like a clean folder.

## Contact

For questions, bug reports, or feature ideas, email [sankhoroy@gmail.com](mailto:sankhoroy@gmail.com).

## License

MIT — see [LICENSE](LICENSE).
