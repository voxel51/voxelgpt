"""
Effective query generator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import pandas as pd

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_cache


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

EFFECTIVE_PROMPT_GENERATOR_PREFIX_PATH = os.path.join(
    PROMPTS_DIR, "effective_prompt_generator_prefix.txt"
)

HISTORY_RELEVANCE_PROMPT_PREFIX_PATH = os.path.join(
    PROMPTS_DIR, "history_relevance_prompt_prefix.txt"
)

HISTORY_RELEVANCE_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "history_relevance_examples.csv"
)


def get_history_relevance_examples():
    cache = get_cache()
    key = "history_relevance_examples"
    if key not in cache:
        df = pd.read_csv(HISTORY_RELEVANCE_EXAMPLES_PATH)
        examples = []

        for _, row in df.iterrows():
            example = {
                "query": row.query,
                "history_is_relevant": row.history_is_relevant,
            }
            examples.append(example)
        cache[key] = examples
    return cache[key]


def load_history_relevance_prompt_prefix():
    cache = get_cache()
    key = "history_relevance_prompt_prefix"
    if key not in cache:
        with open(HISTORY_RELEVANCE_PROMPT_PREFIX_PATH, "r") as f:
            cache[key] = f.read()
    return cache[key]


def generate_history_relevance_prompt(query):
    prefix = load_history_relevance_prompt_prefix()
    history_relevance_examples = get_history_relevance_examples()

    history_relevance_example_formatter_template = """
    Query: {query}
    Is history relevant: {history_is_relevant}
    """

    history_relevance_prompt = PromptTemplate(
        input_variables=["query", "history_is_relevant"],
        template=history_relevance_example_formatter_template,
    )

    history_relevance_prompt = FewShotPromptTemplate(
        examples=history_relevance_examples,
        example_prompt=history_relevance_prompt,
        prefix=prefix,
        suffix="Query: {query}\nIs history relevant: ",
        input_variables=["query"],
        example_separator="\n",
    )

    return history_relevance_prompt.format(query=query)


def load_effective_prompt_prefix_template():
    cache = get_cache()
    key = "effective_prompt_prefix"
    if key not in cache:
        with open(EFFECTIVE_PROMPT_GENERATOR_PREFIX_PATH, "r") as f:
            cache[key] = f.read()
    return cache[key]


def format_chat_history(chat_history):
    return "\n".join(chat_history) + "\n"


def generate_dataset_view_prompt(chat_history):
    prompt = load_effective_prompt_prefix_template()
    prompt += format_chat_history(chat_history)
    prompt += "Effective prompt: "
    return prompt


def _keyword_is_present(query):
    return "now " in query.lower()


def _history_is_relevant(query):
    if _keyword_is_present(query):
        return True

    prompt = generate_history_relevance_prompt(query)
    response = get_llm().call_as_llm(prompt)
    return "yes" in response.strip().lower()


def _process_query(query):
    if query[:6] == "User: ":
        query = query[6:]
    return query


def generate_effective_query(chat_history):
    query = _process_query(chat_history[-1])
    if not _history_is_relevant(query):
        return query

    prompt = generate_dataset_view_prompt(chat_history)
    response = get_llm().call_as_llm(prompt)
    return response.strip()
