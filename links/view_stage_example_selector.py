"""
View stage example selector.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import hashlib
import json
import os

import chromadb
from chromadb.utils import embedding_functions
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

EXAMPLE_EMBEDDINGS_PATH = os.path.join(EXAMPLES_DIR, "embeddings.json")
VIEW_STAGE_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "fiftyone_viewstage_examples.csv"
)

COLLECTION_NAME = "fiftyone_view_stage_examples"

VIEW_STAGE_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

api_key = os.environ.get("OPENAI_API_KEY", None)
if api_key is None:
    raise ValueError(
        "You must provide an OpenAI key by setting the OPENAI_API_KEY "
        "environment variable"
    )

ada_002 = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key, model_name="text-embedding-ada-002"
)

client = chromadb.Client()


def load_chromadb():
    try:
        client.get_collection(COLLECTION_NAME, embedding_function=ada_002)
    except:
        create_chroma_collection()


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
        new_embeddings = ada_002(new_queries)
        for key, embedding in zip(new_hashes, new_embeddings):
            example_embeddings[key] = embedding

    for key in query_hashes:
        query_embeddings.append(example_embeddings[key])

    if new_queries:
        print("Saving embeddings to disk...")
        with open(EXAMPLE_EMBEDDINGS_PATH, "w") as f:
            json.dump(example_embeddings, f)

    return query_embeddings


def create_chroma_collection():
    collection = client.create_collection(
        COLLECTION_NAME, embedding_function=ada_002
    )

    examples = pd.read_csv(VIEW_STAGE_EXAMPLES_PATH, on_bad_lines="skip")
    queries = examples["query"].tolist()
    media_types = examples["media_type"].tolist()
    geos = examples["geo"].tolist()
    geos = [str(int(g)) for g in geos]
    stages_lists = examples["stages"].tolist()
    ids = [f"{i}" for i in range(len(queries))]
    metadatas = [
        {"input": query, "output": sl, "media_type": mt, "geo": geo}
        for query, sl, mt, geo in zip(queries, stages_lists, media_types, geos)
    ]

    embeddings = get_or_create_embeddings(queries)
    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)


def has_geo_field(dataset):
    types = list(dataset.get_field_schema(flat=True).values())
    types = [type(t) for t in types]
    return any(["Geo" in t.__name__ for t in types])


def get_similar_examples(dataset, query):
    load_chromadb()
    collection = client.get_collection(
        COLLECTION_NAME, embedding_function=ada_002
    )

    media_type = dataset.media_type
    geo = has_geo_field(dataset)

    _media_filter = {
        "$or": [
            {"media_type": {"$eq": "all"}},
            {"media_type": {"$eq": media_type}},
        ]
    }

    if geo:
        _filter = _media_filter
    else:
        _filter = {"$and": [_media_filter, {"geo": {"$eq": "0"}}]}

    res = collection.query(
        query_texts=[query], n_results=20, where=_filter, include=["metadatas"]
    )["metadatas"][0]

    similar_examples = [
        {"input": r["input"], "output": r["output"]} for r in res
    ]

    return similar_examples


def generate_view_stage_examples_prompt_template(dataset, query):
    examples = get_similar_examples(dataset, query)
    example_prompt = VIEW_STAGE_EXAMPLE_PROMPT
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Generate code to produce the FiftyOne view stages for the following prompts:\n",
        suffix="Input: {text}\nOutput:",
        input_variables=["text"],
    )


def generate_view_stage_examples_prompt(dataset, query):
    similar_examples_prompt_template = (
        generate_view_stage_examples_prompt_template(dataset, query)
    )
    return similar_examples_prompt_template.format(text=query)
