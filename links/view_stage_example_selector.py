"""
View stage example selector.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import hashlib
import json
import os

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import pandas as pd

# pylint: disable=relative-beyond-top-level
from .utils import get_chromadb_client, get_embedding_function


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

EXAMPLE_EMBEDDINGS_PATH = os.path.join(
    EXAMPLES_DIR,
 "viewstage_embeddings.json")
VIEW_STAGE_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "fiftyone_viewstage_examples.csv"
)

CHROMADB_COLLECTION_NAME = "fiftyone_view_stage_examples"

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
        with open(EXAMPLE_EMBEDDINGS_PATH, "r") as f:
            example_embeddings = json.load(f)
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
        with open(EXAMPLE_EMBEDDINGS_PATH, "w") as f:
            json.dump(example_embeddings, f)

    return query_embeddings

def _format_bool_column(col):
    return [str(int(entry)) for entry in col.tolist()]

def _contains_match_or_filter_labels(stages_list):
    cond = ("match" in stages_list) or ("filter_labels" in stages_list)
    return str(int(cond))

def create_chroma_collection(client):
    collection = client.create_collection(
        CHROMADB_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )

    examples = pd.read_csv(VIEW_STAGE_EXAMPLES_PATH, on_bad_lines="skip")
    queries = examples["query"].tolist()
    media_types = examples["media_type"].tolist()
    geos = _format_bool_column(examples["geo"])
    label_types = examples["label_type"].tolist()
    text_sims = _format_bool_column(examples["text_sim"])
    image_sims = _format_bool_column(examples["image_sim"])
    evals = _format_bool_column(examples["eval"])
    metas = _format_bool_column(examples["metadata"])

    stages_lists = examples["stages"].tolist()
    match_filter = [_contains_match_or_filter_labels(sl) for sl in stages_lists]

    ids = [f"{i}" for i in range(len(queries))]
    metadatas = [
        {
            "input": query, 
            "output": sl, 
            "media_type": mt, 
            "geo": geo,
            "label_type": lt,
            "text_sim": ts,
            "image_sim": ims,
            "eval": eval,
            "meta": meta,
            "match_filter": mf,
            }
        for query, sl, mt, geo, lt, ts, ims, eval, meta, mf in zip(
            queries, 
            stages_lists, 
            media_types, 
            geos,
            label_types,
            text_sims,
            image_sims,
            evals,
            metas,
            match_filter,
            )
    ]

    embeddings = get_or_create_embeddings(queries)
    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    return collection


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
        prompt = prompt.replace(
            "TEXT_SIM_KEY", 
            runs["text_similarity"]['key']
            )
    if "image_similarity" in runs:
        prompt = prompt.replace(
            "IMAGE_SIM_KEY", 
            runs["image_similarity"]['key']
            )
    if "evaluation" in runs:
        prompt = prompt.replace(
            "EVAL_KEY", 
            runs["evaluation"]['key']
            )
    return prompt

def _count_empty_class_names(label_field):
    return [
        list(class_name.values())[0] 
        for class_name in label_field
        ].count([])

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
            key for key, count in zip(label_field_keys, empty_counts)
            if count == min_empty_count
        ]
        return {key: label_fields[key] for key in valid_keys}, min_empty_count

def _parse_runs_and_labels(runs, label_fields):
    reduced_label_fields, count = _reduce_label_fields(label_fields.copy())
    reduced_runs = runs.copy()
    if count is not None and count > 0 and "text_similarity" in reduced_runs:
        reduced_label_fields = None

    return reduced_runs, reduced_label_fields


def _load_examples_vectorstore():
    client = get_chromadb_client()
    try:
        collection = client.get_collection(
            CHROMADB_COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    except:
        collection = create_chroma_collection(client)
    return collection

def initialize_examples_vectorstore():
    examples_db = _load_examples_vectorstore()
    globals()['examples_db'] = examples_db

def get_similar_examples(sample_collection, query, runs, label_fields):
    if 'examples_db' not in globals():
        initialize_examples_vectorstore()
    examples_db = globals()['examples_db']

    red_runs, red_label_fields = _parse_runs_and_labels(runs, label_fields)

    media_type = sample_collection.media_type
    geo = has_geo_field(sample_collection)
    text_sim = "text_similarity" in red_runs
    image_sim = "image_similarity" in red_runs
    meta = "metadata" in red_runs
    eval = "eval" in red_runs

    _filter = {
        "$or": [
            {"media_type": {"$eq": "all"}},
            {"media_type": {"$eq": media_type}},
        ]
    }

    if red_label_fields:
        label_types = list(set(
            [
                get_label_type(sample_collection, field) 
                for field in red_label_fields
                ]
            ))
        label_types.append("all")

        _label_filter_or = [
            {"label_type": {"$eq": lt}}
            for lt in label_types
        ]

        _label_filter = {
            "$or": _label_filter_or
        }
        
        _filter = {"$and": [_filter, _label_filter]}

    def add_and_to_filter(_filter, new_str):
        _filter = {"$and": [_filter, {new_str: {"$eq": "0"}}]}
        return _filter
    
    match_filter = red_label_fields and not text_sim

    conds = [geo, text_sim, image_sim, meta, eval, match_filter]
    strs = ["geo", "text_sim", "image_sim", "meta", "eval", "match_filter"]

    for cond, new_str in zip(conds, strs):
        _filter = add_and_to_filter(_filter, new_str) if not cond else _filter

    res = examples_db.query(
        query_texts=[query], n_results=20, where=_filter, include=["metadatas"]
    )["metadatas"][0]

    similar_examples = [
        {"input": r["input"], "output": r["output"]} for r in res
    ]

    return similar_examples


def generate_view_stage_examples_prompt_template(
        sample_collection, 
        query, 
        runs,
        label_fields
        ):
    examples = get_similar_examples(
        sample_collection, 
        query, 
        runs, 
        label_fields
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
        sample_collection, 
        query,
        runs,
        label_fields
        ):
    similar_examples_prompt_template = (
        generate_view_stage_examples_prompt_template(
        sample_collection, 
        query,
        runs,
        label_fields
        )
    )
    
    prompt = similar_examples_prompt_template.format(text=query)
    prompt = _replace_run_keys(prompt, runs)
    return prompt
