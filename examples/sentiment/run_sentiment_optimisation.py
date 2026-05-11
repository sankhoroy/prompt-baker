# from __future__ import annotations

from pathlib import Path
from typing import Any

from prompt_baker import ChatModelSpec, OptimizerConfig, PromptBakerOptimizer
from prompt_baker.types import CompletionFn
from prompt_baker.visualizer import plot_progress

# from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from prompt_baker import ChatModelSpec, OptimizerConfig, PromptBakerOptimizer
from prompt_baker.types import CompletionFn

from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
import os
import pandas as pd

from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "your-groq-api-key"

def heuristic_completion(system_prompt: str, user_prompt: str) -> str:
    del system_prompt
    text = user_prompt.lower()
    positive_words = {"love", "amazing", "great", "helpful", "inspiring", "recommend"}
    negative_words = {"cold", "tasteless", "regret", "waste", "crashes", "dirty", "terrible", "rude"}
    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)
    return "positive" if pos >= neg else "negative"

def groq_llm_completion(system_prompt: str, user_prompt: str) -> str:

    llm_groq = ChatGroq(
        model="llama-3.3-70b-versatile"
    )
    agent = create_agent(llm_groq,system_prompt=system_prompt)
    
    response = agent.invoke(
        {"messages": [("user", user_prompt )]}
    )
    return response['messages'][-1].content




def main() -> None:
    example_dir = Path(__file__).parent

    models = [
        ChatModelSpec(
            name="heuristic-default",
            completion_fn=heuristic_completion,
        ),
        ChatModelSpec(
            name="model-groq",
            completion_fn=groq_llm_completion
        )
    ]

    config = OptimizerConfig(
        task_type="classification",
        metric="accuracy",
        generations=10,
        population_size=12,
        token_length_optimisation=True,
    )


    config = OptimizerConfig(
        task_type="classification",
        metric="accuracy",
        generations=4,
        population_size=12,
        token_length_optimisation=True,
    )

    optimizer = PromptBakerOptimizer(
        model_specs=models,
        system_prompts=[
            "You are a sentiment classifier. Return only one word: positive or negative.",
            "Classify sentiment in strict format. Output exactly positive or negative.",
        ],
        user_prompts=[
            "Classify sentiment of this text: {input}",
            "Determine if the sentiment is positive or negative.\nText: {input}",
        ],
        config=config,
        run_dir="logs"
    )

    csv_path = "benchmark_sentiment.csv"
    best = optimizer.optimize(example_dir / csv_path,verbose=True)
    print("Best candidate:")
    print(best)
    print(f"Run logs: {optimizer.logger.run_dir}")
    plot_file = plot_progress(optimizer.logger.run_dir)
    print(f"Progress plot: {plot_file}")


if __name__ == "__main__":
    main()
