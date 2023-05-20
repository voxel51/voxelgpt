"""
Query intent classifier.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import pandas as pd

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_moderator


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

QUERY_INTENT_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "fiftyone_query_intent_examples.csv"
)
INTENT_TASK_RULES_PATH = os.path.join(
    PROMPTS_DIR, "intent_classification_task_rules.txt"
)


def load_query_classifier_prefix():
    with open(INTENT_TASK_RULES_PATH, "r") as f:
        return f.read()


def get_query_intent_examples():
    df = pd.read_csv(QUERY_INTENT_EXAMPLES_PATH)
    df = df.sample(frac=1).reset_index(drop=True)
    examples = []

    for _, row in df.iterrows():
        example = {"query": row.query, "intent": row.intent}
        examples.append(example)
    return examples


def generate_query_classifier_prompt(query):
    prefix = load_query_classifier_prefix()
    intent_examples = get_query_intent_examples()

    intent_example_formatter_template = """
    Query: {query}
    Intent: {intent}\n
    """

    classification_prompt = PromptTemplate(
        input_variables=["query", "intent"],
        template=intent_example_formatter_template,
    )

    query_classifier_prompt = FewShotPromptTemplate(
        examples=intent_examples,
        example_prompt=classification_prompt,
        prefix=prefix,
        suffix="Query: {query}\nIntent:",
        input_variables=["query"],
        example_separator="\n",
    )

    return prefix + query_classifier_prompt.format(query=query)


def moderate_query(query):
    return get_moderator().run(query)


def classify_query_intent(query):
    prompt = generate_query_classifier_prompt(query)
    res = get_llm().call_as_llm(prompt).strip()

    if "display" in res:
        return "display"
    elif "documentation" in res:
        return "documentation"
    elif "computer vision" in res:
        return "computer_vision"
    else:
        return "confused"
