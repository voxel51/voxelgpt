"""
Query intent classifier.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import pickle

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_cache, hash_query, get_embedding_function
from .view_stage_example_selector import get_examples


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

NON_VIEWSTAGE_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "query_intent_examples.csv"
)
NON_VIEWSTAGE_EMBEDDINGS_PATH = os.path.join(
    EXAMPLES_DIR, "query_intent_embeddings.pkl"
)
VIEWSTAGE_EXAMPLES_PATH = os.path.join(EXAMPLES_DIR, "viewstage_examples.csv")
VIEWSTAGE_EMBEDDINGS_PATH = os.path.join(
    EXAMPLES_DIR, "viewstage_embeddings.pkl"
)
INTENT_TASK_RULES_PATH = os.path.join(
    PROMPTS_DIR, "intent_classification_rules.txt"
)

DISPLAY_KEYWORDS = (
    "display",
    "show",
)

DOCUMENTATION_KEYWORDS = (
    "docs",
    "documentation",
    "fiftyone",
    "fifty one",
)


def _load_viewstage_examples():
    viewstage_examples, viewstage_embeddings = get_examples()
    viewstage_examples = viewstage_examples[["query"]]
    viewstage_examples = viewstage_examples.assign(intent="display")
    viewstage_examples["hash"] = viewstage_examples["query"].apply(
        lambda x: hash_query(x)
    )
    return viewstage_examples, viewstage_embeddings


def _load_non_viewstage_examples():
    examples = pd.read_csv(NON_VIEWSTAGE_EXAMPLES_PATH, on_bad_lines="skip")
    examples = examples[["query", "intent"]]
    examples["hash"] = examples["query"].apply(lambda x: hash_query(x))

    embeddings = _get_or_create_embeddings(examples)
    return examples, embeddings


def _get_or_create_embeddings(examples):
    if os.path.isfile(NON_VIEWSTAGE_EMBEDDINGS_PATH):
        with open(NON_VIEWSTAGE_EMBEDDINGS_PATH, "rb") as f:
            example_embeddings = pickle.load(f)
    else:
        example_embeddings = {}

    queries = examples["query"].tolist()
    hashes = examples["hash"].tolist()

    new_hashes = []
    new_queries = []

    for hash, query in zip(hashes, queries):
        if hash not in example_embeddings:
            new_hashes.append(hash)
            new_queries.append(query)

    if new_queries:
        print("Generating %d embeddings..." % len(new_queries))
        model = get_embedding_function()
        new_embeddings = model(new_queries)
        for hash, embedding in zip(new_hashes, new_embeddings):
            example_embeddings[hash] = embedding

    if new_queries:
        print("Saving embeddings to disk...")
        with open(NON_VIEWSTAGE_EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(example_embeddings, f)

    return example_embeddings


def _load_examples():
    nv_ex, nv_emb = _load_non_viewstage_examples()
    v_ex, v_emb = _load_viewstage_examples()

    examples = pd.concat([nv_ex, v_ex])
    embeddings = {**nv_emb, **v_emb}

    queries = examples["query"].tolist()
    intents = examples["intent"].tolist()
    hashes = examples["hash"].tolist()
    ordered_embeddings = [np.array(embeddings[key]) for key in hashes]
    return queries, intents, ordered_embeddings


def _get_examples():
    cache = get_cache()
    keys = ("queries", "intents", "embeddings")
    if keys[0] not in cache or keys[1] not in cache or keys[2] not in cache:
        cache[keys[0]], cache[keys[1]], cache[keys[2]] = _load_examples()
    return cache[keys[0]], cache[keys[1]], cache[keys[2]]


def get_similar_examples(query, k=20):
    queries, intents, embeddings = _get_examples()

    model = get_embedding_function()
    query_embedding = np.array(model([query]))

    dists = np.array([cosine(query_embedding, emb) for emb in embeddings])

    sorted_ix = np.argsort(dists).astype(int)

    similar_queries = [queries[ix] for ix in sorted_ix[:k]]
    similar_intents = [intents[ix] for ix in sorted_ix[:k]]

    return [
        {"query": sq, "intent": ss}
        for sq, ss in zip(similar_queries, similar_intents)
    ]


def _assemble_query_intent_prompt(query):
    prefix = _load_query_classifier_prefix()
    examples = get_similar_examples(query)

    intent_example_formatter_template = """
    Query: {query}
    Intent: {intent}\n
    """

    classification_prompt = PromptTemplate(
        input_variables=["query", "intent"],
        template=intent_example_formatter_template,
    )

    template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=classification_prompt,
        prefix=prefix,
        suffix="Query: {query}\nIntent:",
        input_variables=["query"],
        example_separator="\n",
    )

    return template.format(query=query)


def _classify_intent_with_examples(query):
    prompt = _assemble_query_intent_prompt(query)
    res = get_llm().call_as_llm(prompt).strip()
    if "documentation" in res:
        return "documentation"
    elif "computer vision" in res:
        return "computer_vision"
    elif "display" in res:
        if "how to" in query or "how do" in query.lower():
            return "documentation"
        return "display"
    else:
        return "confused"


def _match_display_keywords(query):
    for keyword in DISPLAY_KEYWORDS:
        if keyword in query.lower():
            return True
    return False


def _match_docs_keywords(query):
    for keyword in DOCUMENTATION_KEYWORDS:
        if keyword in query.lower():
            return True
    return False


def _classify_intent_with_keywords(query):
    if _match_display_keywords(query):
        return "display"

    if _match_docs_keywords(query):
        return "documentation"


def _load_query_classifier_prefix():
    cache = get_cache()
    key = "query_classifier_prefix"
    if key not in cache:
        with open(INTENT_TASK_RULES_PATH, "r") as f:
            cache[key] = f.read()
    return cache[key]


def classify_query_intent(query):
    intent = _classify_intent_with_keywords(query)
    if intent:
        return intent

    return _classify_intent_with_examples(query)
