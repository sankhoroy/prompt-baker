"""
Cat-Dog RAG Optimization Script

This script implements a Retrieval-Augmented Generation (RAG) pipeline 
and uses the PromptBaker library to optimize system/user prompts 
and retrieval strategies for a cat/dog related questions.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Callable

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# PromptBaker Imports
from prompt_baker.optimizer import PromptBakerOptimizer
from prompt_baker.types import ChatModelSpec, OptimizerConfig
from prompt_baker.visualizer import plot_progress, create_scores_csv

# --- CONFIGURATION & API KEYS ---
# It is recommended to use environment variables for security
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY","your-groq-api-key")

def setup_data():
    """Creates the local knowledge base and benchmark dataset files."""
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

def get_vectorstore():
    """Initializes the embedding model and creates/loads the Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    raw_text = Path("knowledge_base.txt").read_text()
    documents = [Document(page_content=raw_text)]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./catdog_chroma_db",
    )
    return vectorstore

def make_rag_completion_fn(retriever, llm):
    """
    Factory function to create a completion function for a specific retriever.
    Required by PromptBaker to test different retrieval strategies.
    """
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

def paraphraser_groq_completion_fn(prompt: str) -> str:
    """Uses Groq to paraphrase prompts for the genetic algorithm mutation step."""
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    messages = [
        SystemMessage(content="You are an expert prompt engineer specialized in paraphrasing and optimizing prompts."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()

def llm_judge_fn(expected_answer: str, predicted_answer: str, scale_max: float) -> float:
    """LLM-as-a-judge function to score the predicted answer against the target."""
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    judge_prompt = f"""
Expected Answer: {expected_answer}
Predicted Answer: {predicted_answer}
Score from 0 to {scale_max}.
Evaluation Criteria: factual correctness, semantic similarity, exactness.
Return ONLY the number.
""".strip()

    response = llm.invoke([
        SystemMessage(content="You are a strict evaluator."),
        HumanMessage(content=judge_prompt),
    ])
    
    try:
        return float(response.content.strip())
    except ValueError:
        return 0.0

def main():
    # 1. Setup Data and Vector Store
    setup_data()
    vectorstore = get_vectorstore()
    llm_main = ChatGroq(model="llama-3.3-70b-versatile")

    # 2. Define Retrieval Strategies to test
    retriever_strategies = {
        "similarity_k1": vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}),
        "similarity_k2": vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        "similarity_k4": vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        "mmr_k2": vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 6}),
        "mmr_k4": vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}),
    }

    # 3. Create Model Specs for the Optimizer
    model_specs = [
        ChatModelSpec(name=name, completion_fn=make_rag_completion_fn(ret, llm_main))
        for name, ret in retriever_strategies.items()
    ]

    # 4. Define Initial Prompt Pool
    system_prompts = [
        "Answer only from retrieved context.",
        "You are a RAG QA assistant for cats and dogs.",
        "Use retrieved evidence before answering.",
        "Never hallucinate information.",
        "Provide concise factual answers.",
    ]

    user_prompts = [
        "{input}",
        "Question:\n{input}",
        "Use context and answer:\n{input}",
        "Answer carefully:\n{input}",
    ]

    # 5. Configure Optimizer
    config = OptimizerConfig(
        task_type="generation",
        metric="llm_as_judge",
        population_size=2,
        generations=3,
        mutation_rate=0.7,
        crossover_rate=0.7,
        elite_size=2,
        random_seed=42,
        input_column="input",
        target_column="target",
        token_length_optimisation=True,
        token_penalty_weight=0.05,
        paraphrases_per_prompt=2,
        judge_scale_max=10.0,
    )

    # 6. Initialize and Run Optimization
    optimizer = PromptBakerOptimizer(
        model_specs=model_specs,
        system_prompts=system_prompts,
        user_prompts=user_prompts,
        config=config,
        paraphrase_completion_fn=paraphraser_groq_completion_fn,
        judge_score_fn=llm_judge_fn,
        run_dir="logs_cat_dog_rag_run2"
    )

    print("Starting the baking process...")
    best_candidate = optimizer.optimize(dataset_csv="benchmark.csv", verbose=True)
    print("\nOptimization Complete!")
    print(f"Best Model Strategy: {best_candidate.model_name}")

    # 7. Visualization and Reporting
    plot_progress('logs_cat_dog_rag_run2')
    report_path = "generated_report_cat_dog_rag_run2.csv"
    create_scores_csv('logs_cat_dog_rag_run2/', report_path)
    
    print(f"Progress plot saved to logs_cat_dog_rag_run2/progress.png")
    print(f"Detailed report saved to {report_path}")

if __name__ == "__main__":
    main()
