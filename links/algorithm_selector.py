"""
Algorithm selector.

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

ALGORITHM_EXAMPLES_PATH = os.path.join(EXAMPLES_DIR, "algorithm_examples.csv")
ALGORITHM_SELECTOR_PREFIX_PATH = os.path.join(
    PROMPTS_DIR, "algorithm_selector_prefix.txt"
)

ALGORITHMS = (
    "uniqueness",
    "image_similarity",
    "text_similarity",
    "mistakenness",
    "hardness",
    "evaluation",
    "metadata",
)


def get_algorithm_examples():
    df = pd.read_csv(ALGORITHM_EXAMPLES_PATH)
    examples = []

    for _, row in df.iterrows():
        algorithms_used = [alg for alg in ALGORITHMS if row[alg] == "Y"]
        example = {"query": row.prompt, "algorithms": algorithms_used}
        examples.append(example)
    return examples


def load_algorithm_selector_prefix():
    with open(ALGORITHM_SELECTOR_PREFIX_PATH, "r") as f:
        return f.read()


def generate_algorithm_selector_prompt(query):
    cache = get_cache()
    key = "algorithm_selector_template"
    if key not in cache:
        prefix = load_algorithm_selector_prefix()
        algorithm_examples = get_algorithm_examples()

        algorithm_example_formatter_template = """
        Query: {query}
        Algorithms used: {algorithms}\n
        """

        algorithms_prompt = PromptTemplate(
            input_variables=["query", "algorithms"],
            template=algorithm_example_formatter_template,
        )

        template = FewShotPromptTemplate(
            examples=algorithm_examples,
            example_prompt=algorithms_prompt,
            prefix=prefix,
            suffix="Query: {query}\nAlgorithms used:",
            input_variables=["query"],
            example_separator="\n",
        )
        cache[key] = template
    else:
        template = cache[key]

    return template.format(query=query)


def select_algorithms(query):
    algorithm_selector_prompt = generate_algorithm_selector_prompt(query)
    res = get_llm().call_as_llm(algorithm_selector_prompt)
    return [alg for alg in ALGORITHMS if alg in res]
