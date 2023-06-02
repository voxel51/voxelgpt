import os
import pickle

import numpy as np
from scipy.spatial.distance import cosine

import fiftyone as fo


# pylint: disable=relative-beyond-top-level
from .utils import get_embedding_function, get_cache, hash_query

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

EXAMPLE_EMBEDDINGS_PATH = os.path.join(EXAMPLES_DIR, "schema_embeddings.pkl")
EXAMPLES_PATH = os.path.join(EXAMPLES_DIR, "schema_examples.txt")

THRESHOLD = 0.075
MODEL = get_embedding_function()


def get_view(sample_collection):
    if isinstance(sample_collection, fo.DatasetView):
        return sample_collection

    return sample_collection.view()


def get_dataset(sample_collection):
    if isinstance(sample_collection, fo.DatasetView):
        return sample_collection._root_dataset

    return sample_collection


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

    ordered_embeddings = [example_embeddings[key] for key in query_hashes]
    return ordered_embeddings


def run_name_query(sample_collection):
    name = get_dataset(sample_collection).name
    return f"Your dataset is named `{name}`."


def run_persistent_query(sample_collection):
    persistent = get_dataset(sample_collection).persistent
    persistent_str = "" if persistent else "not "
    return f"Your dataset is {persistent_str}persistent."


def run_dataset_samples_query(sample_collection):
    dataset = get_dataset(sample_collection)
    num_samples = dataset.count()
    return f"Your dataset has `{num_samples}` samples."


def run_view_samples_query(sample_collection):
    view = get_view(sample_collection)
    num_samples = view.count()
    return f"Your view has `{num_samples}` samples."


def run_field_query(sample_collection):
    dataset = get_dataset(sample_collection)
    sample = dataset.first()
    field_names = sample.field_names

    message = "Your dataset has the following fields: "
    for fn in field_names:
        type = sample[fn].__class__.__name__
        message += f"`{fn}`:  `{type}`,"
    return message


def run_tags_query(sample_collection):
    dataset = get_dataset(sample_collection)
    tags = dataset.distinct("tags")
    if len(tags) == 0:
        return "Your dataset has no tags."
    tags = ", ".join([f"`{tag}`" for tag in tags])
    return f"Your dataset has the following tags: {tags}"


def run_brain_runs_query(sample_collection):
    brain_runs = sample_collection.list_brain_runs()
    if len(brain_runs) == 0:
        return "Your dataset has no brain runs."
    brain_runs = ", ".join([f"`{br}`" for br in brain_runs])
    return f"Your dataset has the following brain runs: {brain_runs}"


def run_evaluations_query(sample_collection):
    evaluations = sample_collection.list_evaluations()
    if len(evaluations) == 0:
        return "Your dataset has no evaluations."
    evaluations_str = ", ".join([f"`{eval}`" for eval in evaluations])
    return f"Your dataset has the following evaluations: {evaluations_str}"


def _get_classification_field_names(sample_collection):
    dataset = get_dataset(sample_collection)
    sample = dataset.first()
    field_names = sample.field_names
    classification_field_names = []
    for fn in field_names:
        if type(sample[fn]) == fo.core.labels.Classification:
            classification_field_names.append(fn)
    return classification_field_names


def run_classifications_query(sample_collection):
    classification_field_names = _get_classification_field_names(
        sample_collection
    )
    if len(classification_field_names) == 0:
        return "Your dataset has no classification fields."
    classification_field_names = ", ".join(
        [f"`{fn}`" for fn in classification_field_names]
    )
    return f"Your dataset has the following classification fields: `{classification_field_names}`"


def _get_detection_field_names(sample_collection):
    dataset = get_dataset(sample_collection)
    sample = dataset.first()
    field_names = sample.field_names
    detections_field_names = []
    for fn in field_names:
        if type(sample[fn]) == fo.core.labels.Detections:
            detections_field_names.append(fn)
    return detections_field_names


def run_detections_query(sample_collection):
    detection_field_names = _get_detection_field_names(sample_collection)
    if len(detection_field_names) == 0:
        return "Your dataset has no detection fields."
    detection_field_names = ", ".join(
        [f"`{fn}`" for fn in detection_field_names]
    )
    return f"Your dataset has the following detection fields: `{detection_field_names}`"


def run_schema_query(sample_collection):
    dataset = get_dataset(sample_collection)
    schema = dataset.get_field_schema()
    return f"Your dataset has the following schema:\n ```{schema}```"


def run_voxelgpt_query(sample_collection):
    message = "Hi! I'm VoxelGPT, is your AI assistant for computer vision. \n\nI can help you with the following tasks:\n\n1.Create a filtered view into your dataset.\n2.Understand the FiftyOne documentation.\n3.Become a better computer vision practitioner.\n\n\nFor more details, type `help`."
    return message


FUNC_STR_DICT = {
    "name": run_name_query,
    "persistent": run_persistent_query,
    "dataset_samples": run_dataset_samples_query,
    "view_samples": run_view_samples_query,
    "field": run_field_query,
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


def load_schema_examples():
    with open(EXAMPLES_PATH, "r") as f:
        queries = f.read()

    queries = queries.split("\n")
    prompts = queries[::3]
    funcs = queries[1::3]
    embeddings = get_or_create_embeddings(prompts)
    return zip(prompts, funcs, embeddings)


def query_schema(query, sample_collection):
    query_embedding = np.array(MODEL(query)[0])
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
