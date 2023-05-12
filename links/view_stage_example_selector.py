import chromadb
from chromadb.utils import embedding_functions
import hashlib
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import json
import os
import pandas as pd


client = chromadb.Client()

ada_002 = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
)

COLLECTION_NAME = "fiftyone_view_stage_examples"

VIEW_STAGE_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)


def hash_query(query):
    hash_object = hashlib.md5(query.encode())
    return hash_object.hexdigest()


def load_chroma_db():
    try:
        _ = client.get_collection(COLLECTION_NAME, embedding_function=ada_002)
    except:
        create_chroma_collection()


def get_or_create_embeddings(queries):
    print("Getting or creating embeddings for queries...")

    example_embeddings_file = "examples/embeddings.json"
    if os.path.exists(example_embeddings_file):
        print("Loading embeddings from file...")
        with open(example_embeddings_file, "r") as f:
            example_embeddings = json.load(f)
    else:
        print("No embeddings file found. Creating new embeddings...")
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

    if len(new_hashes) > 0:
        print("Creating new embeddings...")
        new_embeddings = ada_002(new_queries)
        for key, embedding in zip(new_hashes, new_embeddings):
            example_embeddings[key] = embedding

    for key in query_hashes:
        query_embeddings.append(example_embeddings[key])

    print("Saving embeddings to file...")
    with open(example_embeddings_file, "w") as f:
        json.dump(example_embeddings, f)

    return query_embeddings


def create_chroma_collection():
    collection = client.create_collection(
        COLLECTION_NAME, embedding_function=ada_002
    )

    examples = pd.read_csv(
        "examples/fiftyone_viewstage_examples.csv", on_bad_lines="skip"
    )
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
    load_chroma_db()
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
