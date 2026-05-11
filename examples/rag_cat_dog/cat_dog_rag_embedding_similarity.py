"""
Cat–dog RAG example using metric ``embedding_similarity`` with a **custom**
HuggingFace / SentenceTransformer embedding model.

The library default is hard-coded in ``prompt_baker.metrics``. This script
**overrides** the function object bound in ``prompt_baker.optimizer`` (where
``PromptBakerOptimizer._score_candidate`` looks it up), so no edits under
``src/`` are required.

Environment:

- ``PROMPT_BAKER_EMBEDDING_MODEL`` — embedding model id (default:
  ``sentence-transformers/all-mpnet-base-v2``).
- ``GROQ_API_KEY`` — for the RAG completion LLM (same as ``cat_dog_rag.py``).

Run from ``examples/rag_cat_dog`` so paths match ``cat_dog_rag.py``.
"""

from __future__ import annotations

import os
from statistics import mean
from typing import Sequence

import prompt_baker.optimizer as _pb_optimizer

_EMBEDDING_MODEL_ID = os.environ.get(
    "PROMPT_BAKER_EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)
_embedding_model = None


def embedding_similarity_metric_override(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> float:
    """Same contract as ``prompt_baker.metrics.embedding_similarity_metric``, custom encoder."""
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError(
            "This example needs sentence-transformers. "
            "Install with: uv pip install sentence-transformers"
        ) from exc

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_ID)

    true_emb = _embedding_model.encode(list(y_true))
    pred_emb = _embedding_model.encode(list(y_pred))
    sim = cosine_similarity(true_emb, pred_emb)
    diagonal_scores = [float(sim[i][i]) for i in range(len(y_true))]
    return float(mean(diagonal_scores)) if diagonal_scores else 0.0


# ``PromptBakerOptimizer`` uses the name imported into this module, not ``metrics`` directly.
_pb_optimizer.embedding_similarity_metric = embedding_similarity_metric_override

import pandas as pd
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompt_baker.optimizer import PromptBakerOptimizer
from prompt_baker.types import ChatModelSpec, OptimizerConfig
from prompt_baker.visualizer import create_scores_csv, plot_progress

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your-groq-api-key")


def setup_data() -> None:
    knowledge_base_text = """
DOG: Labrador Retriever
Color: Golden
Traits: Friendly, intelligent, family dog.

DOG: German Shepherd
Color: Black and tan
Traits: Loyal, protective, police dog.

DOG: Siberian Husky
Color: Grey and white
Traits: Energetic, wolf-like appearance.

DOG: Poodle
Color: White
Traits: Smart, curly fur.

DOG: Beagle
Color: Brown, black, white
Traits: Small hunting dog.

CAT: Persian Cat
Color: White
Traits: Long fur, calm temperament.

CAT: Russian Blue
Color: Grey
Traits: Quiet, intelligent, short-haired.

CAT: Siamese Cat
Color: Cream with dark brown points
Traits: Vocal, social.

CAT: Bengal Cat
Color: Orange with black spots
Traits: Wild appearance, energetic.

CAT: Maine Coon
Color: Brown tabby
Traits: Large size, fluffy fur.
""".strip()

    Path("knowledge_base.txt").write_text(knowledge_base_text, encoding="utf-8")

    benchmark_data = [
        {"input": "Which cat is grey in colour?", "target": "Russian Blue"},
        {"input": "Which dog is golden in colour?", "target": "Labrador Retriever"},
        {"input": "Which dog looks wolf-like?", "target": "Siberian Husky"},
        {"input": "Which cat is vocal?", "target": "Siamese Cat"},
        {"input": "Which dog has curly fur?", "target": "Poodle"},
        {"input": "Which cat has black spots?", "target": "Bengal Cat"},
    ]
    pd.DataFrame(benchmark_data).to_csv("benchmark.csv", index=False)


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    raw_text = Path("knowledge_base.txt").read_text()
    documents = [Document(page_content=raw_text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(documents)
    return Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./catdog_chroma_db",
    )


def make_rag_completion_fn(retriever, llm):
    def rag_completion_fn(system_prompt: str, user_prompt: str) -> str:
        docs = retriever.invoke(user_prompt)
        context = "\n\n".join(doc.page_content for doc in docs)
        final_prompt = f"""
Context:
{context}

{system_prompt}

Question:
{user_prompt}
""".strip()
        response = llm.invoke(final_prompt)
        return response.content.strip()

    return rag_completion_fn


def main() -> None:
    print(f"Embedding metric model (override): {_EMBEDDING_MODEL_ID}\n")

    setup_data()
    vectorstore = get_vectorstore()
    llm_main = ChatGroq(model="llama-3.3-70b-versatile")

    retriever_strategies = {
        "similarity_k2": vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        ),
        "mmr_k2": vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 6},
        ),
    }

    model_specs = [
        ChatModelSpec(name=name, completion_fn=make_rag_completion_fn(ret, llm_main))
        for name, ret in retriever_strategies.items()
    ]

    system_prompts = [
        "Answer only from retrieved context. Reply with the breed name only.",
        "You are a RAG QA assistant. Give the shortest correct breed name.",
    ]
    user_prompts = [
        "{input}",
        "Question:\n{input}",
    ]

    run_dir = "logs_cat_dog_embedding_similarity"
    config = OptimizerConfig(
        task_type="generation",
        metric="embedding_similarity",
        population_size=4,
        generations=3,
        mutation_rate=0.5,
        crossover_rate=0.6,
        elite_size=2,
        random_seed=42,
        token_length_optimisation=True,
        token_penalty_weight=0.05,
        paraphrases_per_prompt=0,
    )

    optimizer = PromptBakerOptimizer(
        model_specs=model_specs,
        system_prompts=system_prompts,
        user_prompts=user_prompts,
        config=config,
        run_dir=run_dir,
        paraphrase_cache_file="paraphrase_cache_embedding.json",
    )

    best = optimizer.optimize(dataset_csv="benchmark.csv", verbose=True)
    print("\nOptimization complete.")
    print(f"Best retriever / prompt combo: {best.model_name}")

    plot_progress(run_dir)
    report_path = "generated_report_cat_dog_embedding_similarity.csv"
    create_scores_csv(f"{run_dir}/", report_path)
    print(f"Plot: {run_dir}/progress.png")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
