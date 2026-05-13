# # Imports

# =========================
# Install dependencies
# =========================

# !pip install -q prompt-baker
# !pip install -q langchain langgraph langchain-core langchain-groq
# !pip install -q pandas matplotlib

# # Optional but useful in Colab
# !pip install -q python-dotenv

from __future__ import annotations
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



os.environ["GROQ_API_KEY"] = "groq-api-key"

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


# ### List down all models with their completion functions


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


# configure Genetic Algorithm optimizer, its population size, if you are not sure keep
# it default.

config = OptimizerConfig(
    task_type="classification",
    metric="f1_score",
    generations=10,
    population_size=12,
    token_length_optimisation=True,
)



# ### Initiate optimiser and ready to start "baking of the raw prompt"


from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage



def paraphraser__groq_completion_fn(prompt: str) -> str:
    llm_groq_paraphraser = ChatGroq(
        model="llama-3.3-70b-versatile"
    )
    response = llm_groq_paraphraser.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert prompt engineer specialized "
                    "in paraphrasing and optimizing prompts."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    return response.content.strip()



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
    paraphrase_completion_fn=paraphraser__groq_completion_fn,
    config=config,
    run_dir="logs"
)


csv_path = "benchmark_sentiment.csv"

best = optimizer.optimize(csv_path,verbose=True)
print("Best candidate:")
print(best)
print(f"Run logs: {optimizer.logger.run_dir}")


# ### Note: While baking in progress keep on checking how much prompt is cooked. `Keep on
# copy the log file some time and transfer it in local and see how much cooking is done`

# # Visualisation

# #### Now while code is running in any server take log files and in local copy log files
# from `run_dir="logs"` in `optimizer` and run this code for Visualisation `plot_progress`

from prompt_baker.visualizer import plot_progress


plot_file = plot_progress(optimizer.logger.run_dir)
print(f"Progress plot: {plot_file}")

