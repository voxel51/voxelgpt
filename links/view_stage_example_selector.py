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

EXAMPLE_EMBEDDINGS_PATH = os.path.join(EXAMPLES_DIR, "embeddings.json")
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
            }
        for query, sl, mt, geo, lt, ts, ims, eval, meta in zip(
            queries, 
            stages_lists, 
            media_types, 
            geos,
            label_types,
            text_sims,
            image_sims,
            evals,
            metas,
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

def get_similar_examples(sample_collection, query, runs, label_fields):
    client = get_chromadb_client()

    try:
        collection = client.get_collection(
            CHROMADB_COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    except:
        collection = create_chroma_collection(client)

    media_type = sample_collection.media_type
    geo = has_geo_field(sample_collection)
    text_sim = "text_similarity" in runs
    image_sim = "image_similarity" in runs
    meta = "metadata" in runs
    eval = "eval" in runs

    _filter = {
        "$or": [
            {"media_type": {"$eq": "all"}},
            {"media_type": {"$eq": media_type}},
        ]
    }

    if label_fields:
        label_types = list(set(
            [
                get_label_type(sample_collection, field) 
                for field in label_fields
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


    if not geo:
        _filter = {"$and": [_filter, {"geo": {"$eq": "0"}}]}

    if not text_sim:
        _filter = {"$and": [_filter, {"text_sim": {"$eq": "0"}}]}
        
    if not image_sim:
        _filter = {"$and": [_filter, {"image_sim": {"$eq": "0"}}]}

    if not meta:
        _filter = {"$and": [_filter, {"meta": {"$eq": "0"}}]}

    if not eval:
        _filter = {"$and": [_filter, {"eval": {"$eq": "0"}}]}

    res = collection.query(
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
