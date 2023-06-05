"""
Dataset schema handler.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import pickle

import numpy as np
from scipy.spatial.distance import cosine
from tabulate import tabulate

import fiftyone as fo

# pylint: disable=relative-beyond-top-level
from .utils import get_embedding_function, get_cache, hash_query


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

EXAMPLE_EMBEDDINGS_PATH = os.path.join(EXAMPLES_DIR, "schema_embeddings.pkl")
EXAMPLES_PATH = os.path.join(EXAMPLES_DIR, "schema_examples.txt")

THRESHOLD = 0.075

SAMPLE_COLLECTION_MESSAGE = "You must provide a sample collection in order for me to respond to this query"

VOXELGPT_MESSAGE = """
Hi! I'm VoxelGPT, your AI assistant for computer vision. I can help you with the following tasks:

1. Create views into your dataset
2. Search the FiftyOne documentation for answers/links
3. Answer general machine learning and computer vision questions

For more information, type 'help'.
"""


def get_view(sample_collection):
    if isinstance(sample_collection, fo.DatasetView):
        return sample_collection

    return sample_collection.view()


def get_dataset(sample_collection):
    if isinstance(sample_collection, fo.DatasetView):
        return sample_collection._root_dataset

    return sample_collection


def load_schema_examples():
    cache = get_cache()
    key = "schema_examples"
    if key not in cache:
        cache[key] = _load_schema_examples()
    return cache[key]


def _load_schema_examples():
    with open(EXAMPLES_PATH, "r") as f:
        queries = f.read()

    queries = queries.split("\n")
    prompts = queries[::3]
    funcs = queries[1::3]
    embeddings = get_or_create_embeddings(prompts)
    return list(zip(prompts, funcs, embeddings))


def get_or_create_embeddings(queries):
    if os.path.isfile(EXAMPLE_EMBEDDINGS_PATH):
        with open(EXAMPLE_EMBEDDINGS_PATH, "rb") as f:
            example_embeddings = pickle.load(f)
    else:
        example_embeddings = {}

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
        model = get_embedding_function()
        new_embeddings = model(new_queries)
        for key, embedding in zip(new_hashes, new_embeddings):
            example_embeddings[key] = embedding

    if new_queries:
        print("Saving embeddings to disk...")
        with open(EXAMPLE_EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(example_embeddings, f)

    return [example_embeddings[key] for key in query_hashes]


def run_name_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    name = get_dataset(sample_collection).name
    return f"Your dataset's name is `{name}`"


def run_persistent_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    dataset = get_dataset(sample_collection)
    persistent_str = "" if dataset.persistent else "not "
    return f"Your dataset is {persistent_str}`persistent`"


def run_dataset_samples_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    dataset = get_dataset(sample_collection)
    num_samples = dataset.count()
    return f"Your dataset has `{num_samples}` samples"


def run_view_samples_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    view = get_view(sample_collection)
    num_samples = view.count()
    return f"Your view has `{num_samples}` samples"


def run_field_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    dataset = get_dataset(sample_collection)
    schema = dataset.get_field_schema()

    message = "Your dataset has the following fields:"
    for field_name, field in schema.items():
        message += f"\n- `{field_name}`: `{type(field).__name__}`"

    return message


def run_tags_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    dataset = get_dataset(sample_collection)
    tags = dataset.distinct("tags")
    if not tags:
        return "Your dataset has no tags"

    tags = "\n".join([f"- `{tag}`" for tag in tags])
    return f"Your dataset has the following tags:\n{tags}"


def run_brain_runs_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    brain_runs = sample_collection.list_brain_runs()
    if not brain_runs:
        return "Your dataset has no brain runs"

    brain_runs = "\n".join([f"- `{br}`" for br in brain_runs])
    return f"Your dataset has the following brain runs:\n{brain_runs}"


def run_evaluations_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    evaluations = sample_collection.list_evaluations()
    if not evaluations:
        return "Your dataset has no evaluation runs"

    evaluations = "\n".join([f"- `{e}`" for e in evaluations])
    return f"Your dataset has the following evaluations:\n{evaluations}"


def run_classifications_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    classification_field_names = _get_classification_field_names(
        sample_collection
    )
    if not classification_field_names:
        return "Your dataset has no classification fields"

    classification_field_names = "\n".join(
        [f"- `{fn}`" for fn in classification_field_names]
    )
    return f"Your dataset has the following classification fields:\n{classification_field_names}"


def _get_classification_field_names(sample_collection):
    dataset = get_dataset(sample_collection)
    sample = dataset.first()
    field_names = sample.field_names
    classification_field_names = []
    for fn in field_names:
        if type(sample[fn]) == fo.core.labels.Classification:
            classification_field_names.append(fn)

    return classification_field_names


def run_detections_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    detection_field_names = _get_detection_field_names(sample_collection)
    if not detection_field_names:
        return "Your dataset has no detection fields"

    detection_field_names = "\n".join(
        [f"- `{fn}`" for fn in detection_field_names]
    )
    return f"Your dataset has the following detection fields:\n{detection_field_names}"


def _get_detection_field_names(sample_collection):
    dataset = get_dataset(sample_collection)
    sample = dataset.first()
    field_names = sample.field_names
    detections_field_names = []
    for fn in field_names:
        if type(sample[fn]) == fo.Detections:
            detections_field_names.append(fn)

    return detections_field_names


def run_schema_query(sample_collection):
    if sample_collection is None:
        return SAMPLE_COLLECTION_MESSAGE

    dataset = get_dataset(sample_collection)
    schema = dataset.get_field_schema()

    # Use this once a verison of FO with `remark-gfm` enabled is shipped
    # https://remarkjs.github.io/react-markdown
    """
    table_str = tabulate(
        list(schema.items()),
        headers=["name", "type"],
        tablefmt="github",
    )
    """

    schema_str = fo.pformat({k: str(v) for k, v in schema.items()})
    table_str = f"```text\n{schema_str}\n```"

    return f"Your dataset has the following schema:\n\n{table_str}"


def run_voxelgpt_query(_):
    return VOXELGPT_MESSAGE.strip()


FUNC_STR_DICT = {
    "name": run_name_query,
    "persistent": run_persistent_query,
    "dataset_samples": run_dataset_samples_query,
    "view_samples": run_view_samples_query,
    "fields": run_field_query,
    "tags": run_tags_query,
    "brain_runs": run_brain_runs_query,
    "evaluations": run_evaluations_query,
    "classifications": run_classifications_query,
    "detections": run_detections_query,
    "schema": run_schema_query,
    "voxelgpt": run_voxelgpt_query,
}


def _run_schema_query(func_str, sample_collection):
    run_func = FUNC_STR_DICT[func_str]
    return run_func(sample_collection)


def query_schema(query, sample_collection):
    model = get_embedding_function()
    query_embedding = np.array(model(query)[0])
    schema_examples = load_schema_examples()

    dist_results = [
        (prompt, func, cosine(query_embedding, embedding))
        for prompt, func, embedding in schema_examples
    ]

    dist_results = sorted(dist_results, key=lambda x: x[2])

    if dist_results[0][2] < THRESHOLD:
        return _run_schema_query(dist_results[0][1], sample_collection)
    else:
        return None
