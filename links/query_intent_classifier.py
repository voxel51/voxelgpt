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
from .utils import get_llm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

QUERY_INTENT_STAGE_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "query_intent_stages_examples.csv"
)
QUERY_INTENT_QA_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "query_intent_qa_examples.csv"
)
INTENT_DISPLAY_TASK_RULES_PATH = os.path.join(
    PROMPTS_DIR, "intent_classification_stages_rules.txt"
)
INTENT_QA_TASK_RULES_PATH = os.path.join(
    PROMPTS_DIR, "intent_classification_qa_rules.txt"
)

DISPLAY_KEYWORDS = (
    "display",
    "show",
)

def _load_prefix(path):
    with open(path, "r") as f:
        return f.read()


def _load_query_classifier_prefix(type):
    if type == "display":
        return _load_prefix(INTENT_DISPLAY_TASK_RULES_PATH)
    else:
        return _load_prefix(INTENT_QA_TASK_RULES_PATH)


def _get_examples(path):
    df = pd.read_csv(path)
    df = df.sample(frac=1).reset_index(drop=True)
    examples = []

    for _, row in df.iterrows():
        example = {"query": row.query, "intent": row.intent}
        examples.append(example)
    return examples


def _get_query_intent_examples(type):
    if type == "display":
        return _get_examples(QUERY_INTENT_STAGE_EXAMPLES_PATH)
    else:
        return _get_examples(QUERY_INTENT_QA_EXAMPLES_PATH)


def _get_query_classifier_prompt_template(type):
    var = 'template_' + type
    if var in globals():
        return globals()[var]
    else:
        prefix = _load_query_classifier_prefix(type)
        intent_examples = _get_query_intent_examples(type)

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

        globals()[var] = template
        return template
    

def _assemble_query_intent_classifier_prompt(query, type):
    template = _get_query_classifier_prompt_template(type)
    return template.format(query=query)

def _match_display_keywords(query):
    for keyword in DISPLAY_KEYWORDS:
        if keyword in query:
            return True
    return False

def classify_query_intent_stages(query):
    if _match_display_keywords(query):
        return "display"
    prompt = _assemble_query_intent_classifier_prompt(query, "display")
    res = get_llm().call_as_llm(prompt).strip()
    if 'display' in res or 'object' in res or 'description' in res:
        return "display"
    else:
        return "confused"
    

def classify_query_intent_qa(query):
    prompt = _assemble_query_intent_classifier_prompt(query, "qa")
    res = get_llm().call_as_llm(prompt).strip()
    if "documentation" in res:
        return "documentation"
    elif "computer vision" in res:
        return "computer_vision"
    else:
        return "confused"


def classify_query_intent(query):
    intent = classify_query_intent_stages(query)
    if intent == "display":
        return intent
    else:
        return classify_query_intent_qa(query)