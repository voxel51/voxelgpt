"""
View stage example selector.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import hashlib
import numpy as np
import os
import pickle
from scipy.spatial.distance import cosine

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import pandas as pd

# pylint: disable=relative-beyond-top-level
from .utils import get_embedding_function, get_cache


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

EXAMPLE_EMBEDDINGS_PATH = os.path.join(
    EXAMPLES_DIR, "viewstage_embeddings.pkl"
)
VIEW_STAGE_EXAMPLES_PATH = os.path.join(EXAMPLES_DIR, "viewstage_examples.csv")

# CHROMADB_COLLECTION_NAME = "fiftyone_view_stage_examples"

VIEW_STAGE_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)


def hash_query(query):
    hash_object = hashlib.md5(query.encode())
    return hash_object.hexdigest()


def get_or_create_embeddings(queries):
    if os.path.isfile(EXAMPLE_EMBEDDINGS_PATH):
        print("Loading embeddings from disk...")
        with open(EXAMPLE_EMBEDDINGS_PATH, "rb") as f:
            example_embeddings = pickle.load(f)
    else:
        example_embeddings = {}

    query_embeddings = []
    query_hashes = []
    new_hashes = []
    new_queries = []

    for query in queries:
        key = hash_query(query)
        query_hashes.append(key)

        if key not in example_embeddings:
            new_hashes.append(key)
            new_queries.append(query)

    if new_queries:
        print("Generating %d embeddings..." % len(new_queries))
        new_embeddings = get_embedding_function()(new_queries)
        for key, embedding in zip(new_hashes, new_embeddings):
            example_embeddings[key] = embedding

    for key in query_hashes:
        query_embeddings.append(example_embeddings[key])

    if new_queries:
        print("Saving embeddings to disk...")

        with open(EXAMPLE_EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(example_embeddings, f)

    return query_embeddings


def has_geo_field(sample_collection):
    types = list(sample_collection.get_field_schema(flat=True).values())
    types = [type(t) for t in types]
    return any(["Geo" in t.__name__ for t in types])


def get_label_type(sample_collection, field_name):
    sample = sample_collection.first()
    field = sample.get_field(field_name)
    field_type = str(type(field).__name__).lower()
    field_type = field_type[:-1] if field_type.endswith("s") else field_type
    return field_type


def _replace_run_keys(prompt, runs):
    if "text_similarity" in runs:
        prompt = prompt.replace("TEXT_SIM_KEY", runs["text_similarity"]["key"])
    if "image_similarity" in runs:
        prompt = prompt.replace(
            "IMAGE_SIM_KEY", runs["image_similarity"]["key"]
        )
    if "evaluation" in runs:
        prompt = prompt.replace("EVAL_KEY", runs["evaluation"]["key"])
    if "uniqueness" in runs:
        prompt = prompt.replace(
            "UNIQUENESS_FIELD", runs["uniqueness"]["uniqueness_field"]
        )
    return prompt


def _count_empty_class_names(label_field):
    return [list(class_name.values())[0] for class_name in label_field].count(
        []
    )


def _reduce_label_fields(label_fields):
    label_field_keys = list(label_fields.keys())
    if len(label_field_keys) == 0:
        return None, None
    elif len(label_field_keys) > 0:
        empty_counts = [
            _count_empty_class_names(label_fields[key])
            for key in label_field_keys
        ]
        min_empty_count = min(empty_counts)
        valid_keys = [
            key
            for key, count in zip(label_field_keys, empty_counts)
            if count == min_empty_count
        ]
        return {key: label_fields[key] for key in valid_keys}, min_empty_count


def _parse_runs_and_labels(runs, label_fields):
    reduced_label_fields, count = _reduce_label_fields(label_fields.copy())
    reduced_runs = runs.copy()
    if count is not None and count > 0 and "text_similarity" in reduced_runs:
        reduced_label_fields = None

    return reduced_runs, reduced_label_fields


def _get_evaluation_type(sample_collection, eval_key):
    eval_cls = sample_collection.get_evaluation_info(eval_key).config.cls
    if "openimages" in eval_cls:
        return "detection"
    elif "coco" in eval_cls:
        return "detection"
    elif "activitynet" in eval_cls:
        return "detection"
    elif "classification" in eval_cls:
        return "classification"
    return None


def _load_examples():
    examples = pd.read_csv(VIEW_STAGE_EXAMPLES_PATH, on_bad_lines="skip")
    examples["meta"] = examples["metadata"]
    examples["contains_match"] = examples["stages"].str.contains("match\(")
    examples["contains_filter_labels"] = examples["stages"].str.contains(
        "filter_labels\("
    )
    examples["mfl"] = (
        examples["contains_match"] | examples["contains_filter_labels"]
    )
    examples["hash"] = examples["query"].apply(lambda x: hash_query(x))

    with open(EXAMPLE_EMBEDDINGS_PATH, "rb") as f:
        embeddings = pickle.load(f)

    embeddings = {
        key: np.array(embeddings[key]) for key in examples["hash"].tolist()
    }
    return examples, embeddings


def get_examples():
    cache = get_cache()
    keys = ("viewstage_examples", "viewstage_embeddings")
    if keys[0] not in cache or keys[1] not in cache:
        cache[keys[0]], cache[keys[1]] = _load_examples()
    return cache[keys[0]], cache[keys[1]]


def _get_filtered_examples(sample_collection, runs, label_fields):
    examples, embeddings = get_examples()
    media_type = sample_collection.media_type
    _filter = examples["media_type"].isin([media_type, "all"])

    red_runs, red_label_fields = _parse_runs_and_labels(runs, label_fields)
    geo = has_geo_field(sample_collection)
    text_sim = "text_similarity" in red_runs
    image_sim = "image_similarity" in red_runs
    meta = "metadata" in red_runs
    eval = "evaluation" in red_runs

    if red_label_fields or eval:
        if red_label_fields:
            label_field_types = list(
                set(
                    [
                        get_label_type(sample_collection, field)
                        for field in red_label_fields
                    ]
                )
            )
        else:
            label_field_types = []

        if eval:
            eval_key = red_runs["evaluation"]["key"]
            eval_types = [_get_evaluation_type(sample_collection, eval_key)]
        else:
            eval_types = []

        label_types = list(set(label_field_types + eval_types + ["all"]))
        _filter = _filter & examples["label_type"].isin(label_types)

    ## contains match() or filter_labels() in stages
    mfl_cond = red_label_fields and not text_sim

    conds = [geo, text_sim, image_sim, meta, eval, mfl_cond]
    strs = ["geo", "text_sim", "image_sim", "meta", "eval", "mfl"]

    for cond, cond_str in zip(conds, strs):
        if not cond:
            _filter = _filter & (examples[cond_str] == False)

    filtered_examples = examples[_filter]

    filtered_queries, filtered_stages, hashes = (
        filtered_examples["query"].tolist(),
        filtered_examples["stages"].tolist(),
        filtered_examples["hash"].tolist(),
    )
    filtered_embeddings = [embeddings[key] for key in hashes]

    return filtered_queries, filtered_stages, filtered_embeddings


def get_similar_examples(sample_collection, query, runs, label_fields):
    ex_queries, ex_stages, ex_embeddings = _get_filtered_examples(
        sample_collection, runs, label_fields
    )

    query_embedding = np.array(get_embedding_function()([query]))

    dists = np.array([cosine(query_embedding, emb) for emb in ex_embeddings])

    sorted_ix = np.argsort(dists).astype(int)

    k = 20
    similar_queries = [ex_queries[ix] for ix in sorted_ix[:k]]
    similar_stages = [ex_stages[ix] for ix in sorted_ix[:k]]

    return [
        {"input": sq, "output": ss}
        for sq, ss in zip(similar_queries, similar_stages)
    ]


def generate_view_stage_examples_prompt_template(
    sample_collection, query, runs, label_fields
):
    examples = get_similar_examples(
        sample_collection, query, runs, label_fields
    )
    example_prompt = VIEW_STAGE_EXAMPLE_PROMPT
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Generate code to produce the FiftyOne view stages for the following prompts:\n",
        suffix="Input: {text}\nOutput:",
        input_variables=["text"],
    )


def generate_view_stage_examples_prompt(
    sample_collection, query, runs, label_fields
):
    similar_examples_prompt_template = (
        generate_view_stage_examples_prompt_template(
            sample_collection, query, runs, label_fields
        )
    )

    prompt = similar_examples_prompt_template.format(text=query)
    prompt = _replace_run_keys(prompt, runs)
    return prompt
