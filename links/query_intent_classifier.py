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
from .utils import get_llm, get_cache


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

QUERY_INTENT_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "fiftyone_query_intent_examples.csv"
)
INTENT_TASK_RULES_PATH = os.path.join(
    PROMPTS_DIR, "intent_classification_task_rules.txt"
)

class QueryIntentClassifier(object):
    """container for query intent classification data."""
    def __init__(self):
        self.template = None


def _load_query_classifier_prefix():
    with open(INTENT_TASK_RULES_PATH, "r") as f:
        return f.read()


def _get_query_intent_examples():
    df = pd.read_csv(QUERY_INTENT_EXAMPLES_PATH)
    df = df.sample(frac=1).reset_index(drop=True)
    examples = []

    for _, row in df.iterrows():
        example = {"query": row.query, "intent": row.intent}
        examples.append(example)
    return examples


def _get_or_create_query_classifier_prompt_template():
    cache = get_cache()
    key = "query_intent_classifier_prompt_template"
    if key in cache:
        return cache[key]
    else:
        prefix = _load_query_classifier_prefix()
        intent_examples = _get_query_intent_examples()

        intent_example_formatter_template = """
        Query: {query}
        Intent: {intent}\n
        """

        classification_prompt = PromptTemplate(
            input_variables=["query", "intent"],
            template=intent_example_formatter_template,
        )

        template = FewShotPromptTemplate(
            examples=intent_examples,
            example_prompt=classification_prompt,
            prefix=prefix,
            suffix="Query: {query}\nIntent:",
            input_variables=["query"],
            example_separator="\n",
        )

        cache[key] = template
        return template
    

def _assemble_query_intent_classifier_prompt(query):
    template = _get_or_create_query_classifier_prompt_template()
    return template.format(query=query)


def classify_query_intent(query):
    prompt = _assemble_query_intent_classifier_prompt(query)
    res = get_llm().call_as_llm(prompt).strip()

    if "display" in res:
        return "display"
    elif "documentation" in res:
        return "documentation"
    elif "computer vision" in res:
        return "computer_vision"
    else:
        return "confused"
