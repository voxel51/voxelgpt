"""
Computation Engine

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
from typing import Optional, Literal

from langchain_core.pydantic_v1 import BaseModel, Field

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.operators as foo
import fiftyone.plugins as fop

# pylint: disable=relative-beyond-top-level
from .utils import (
    PROMPTS_DIR,
    _build_custom_chain,
    _build_chat_chain,
    gpt_3_5,
    gpt_4o,
    get_prompt_from,
)

SHOULD_COMPUTE_CLASSIFICATION_PATH = os.path.join(
    PROMPTS_DIR, "should_run_computation_classification.txt"
)

DELEGATE_COMPUTATION_PATH = os.path.join(
    PROMPTS_DIR, "delegate_computation.txt"
)


def should_run_computation(query):
    prompt = get_prompt_from(SHOULD_COMPUTE_CLASSIFICATION_PATH).format(
        query=query
    )
    intent_chain = _build_custom_chain(gpt_3_5, prompt=prompt)

    topic = intent_chain.invoke({"query": query}).lower()
    return "compute" in topic


def delegate_computation(query):
    prompt = get_prompt_from(DELEGATE_COMPUTATION_PATH).format(query=query)
    intent_chain = _build_custom_chain(gpt_4o, prompt=prompt)
    allowed_topics = (
        "brightness",
        "entropy",
        "uniqueness",
        "similarity",
        "dimensionality reduction",
        "clustering",
        "duplicates",
    )

    topic = intent_chain.invoke({"query": query}).lower()
    for allowed_topic in allowed_topics:
        if allowed_topic in topic:
            return allowed_topic
    return "other"


def run_computation(dataset, assignee, query):
    if assignee == "brightness":
        return compute_brightness(dataset)
    elif assignee == "entropy":
        return compute_entropy(dataset)
    elif assignee == "uniqueness":
        return compute_uniqueness(dataset)
    elif assignee == "similarity":
        return compute_similarity(dataset)
    elif assignee == "dimensionality reduction":
        return compute_dimensionality_reduction(dataset, query)
    elif assignee == "clustering":
        return compute_clustering(dataset)
    elif assignee == "duplicates":
        return compute_duplicates(dataset)
    else:
        return "Computation not implemented yet."


def computation_is_possible(assignee):
    if assignee == "brightness":
        return _can_compute_brightness()
    elif assignee == "entropy":
        return _can_compute_entropy()
    elif assignee == "clustering":
        return _can_compute_clustering()
    else:
        return True


def computation_failure_message(assignee):
    if assignee == "brightness":
        return _brightness_failure_message
    elif assignee == "entropy":
        return _entropy_failure_message
    elif assignee == "clustering":
        return _clustering_failure_message
    else:
        return "Computation not implemented yet."


def computation_already_done(dataset, assignee):
    if assignee == "brightness":
        return dataset.has_field("brightness")
    elif assignee == "entropy":
        return dataset.has_field("entropy")
    elif assignee == "uniqueness":
        return dataset.has_field("uniqueness")
    elif assignee == "duplicates":
        return dataset.has_field("duplicate_key")
    elif assignee == "similarity":
        return any(
            key.startswith("clip_sim") for key in dataset.list_brain_runs()
        )
    ##! Add more checks here for clustering and dimensionality reduction, but need more info
    return False


def compute_dimensionality_reduction(dataset, query, *args, **kwargs):
    prompt_path = os.path.join(PROMPTS_DIR, "compute_visualization.txt")
    prompt = get_prompt_from(prompt_path).format(query=query)
    output_type = DimensionalityReduction

    chain = _build_chat_chain(gpt_4o, prompt=prompt, output_type=output_type)
    dim_red = chain.invoke({"messages": [("user", query)]})

    method = dim_red.method if dim_red.method else "umap"
    key = dim_red.brain_key
    if not key:
        key = _generate_new_key(dataset, f"{method}_vis")

    vis_kwargs = dict(method=method, brain_key=key)
    embedding_fields = list(
        dataset.get_field_schema(ftype=fo.VectorField).keys()
    )
    if not embedding_fields:
        model = "clip-vit-base32-torch"
        message = (
            "No embeddings found. Computing embeddings using CLIP model.\n"
        )
        vis_kwargs["model"] = model
    else:
        embedding_field = embedding_fields[0]
        message = (
            f"Embeddings found in the dataset. Using `{embedding_field}`\n"
        )
        vis_kwargs["embedding_field"] = embedding_field
    fob.compute_visualization(dataset, **vis_kwargs)
    message += f"Visualization computed successfully. A new brain key `{key}` has been added to the dataset. Refresh the dataset to view the visualization."
    return message


def compute_similarity(dataset, *args, **kwargs):
    ## Check for a CLIP field
    candidate_fields = dataset.get_field_schema(ftype=fo.VectorField)
    clip_fields = [
        field for field in candidate_fields.keys() if "clip" in field.lower()
    ]
    sim_kwargs = dict(model="clip-vit-base32-torch")
    if not clip_fields:
        message = "No CLIP embeddings found in the dataset. Computing embeddings first.\n"
    else:
        clip_field = clip_fields[0]
        message = (
            f"CLIP embeddings found in the dataset. Using `{clip_field}`\n"
        )
        sim_kwargs["embeddings"] = clip_field

    key = _generate_new_key(dataset, "clip_sim")
    sim_kwargs["brain_key"] = key
    fob.compute_similarity(dataset, **sim_kwargs)
    message += f"Similarity computed successfully. A new brain key `{key}` has been added to the dataset. Refresh the dataset to view the similarity information."


def compute_uniqueness(dataset, *args, **kwargs):
    fob.compute_uniqueness(dataset)
    return "Uniqueness computed successfully. A new field `uniqueness` has been added to the dataset. Refresh the dataset to view the uniqueness information."


def compute_duplicates(dataset, *args, **kwargs):
    res = fob.compute_exact_duplicates(dataset)
    if not res:
        return "No duplicates found."
    dataset.add_sample_field("duplicate_key", fo.StringField)
    for k, v in res.items():
        dups = dataset.select(v + [k])
        dups.set_values("duplicate_key", [str(k)] * (len(v) + 1))
        dups.save("duplicate_key")
    return "Duplicates computed successfully. A new field `duplicate_key` has been added to the dataset. Refresh the dataset to view the duplicate information."


def compute_brightness(dataset, *args, **kwargs):
    uri = "@jacobmarks/image_issues/compute_brightness"
    ctx = dict(dataset=dataset)
    params = dict(patches_field=None, delegate=False)
    foo.execute_operator(uri, ctx, params=params)
    return "Brightness computed successfully. A new field `brightness` has been added to the dataset. Refresh the dataset to view the brightness information."


def compute_entropy(dataset, *args, **kwargs):
    uri = "@jacobmarks/image_issues/compute_entropy"
    ctx = dict(dataset=dataset)
    params = dict(patches_field=None, delegate=False)
    foo.execute_operator(uri, ctx, params=params)
    return "Entropy computed successfully. A new field `entropy` has been added to the dataset. Refresh the dataset to view the entropy information."


def compute_clustering(dataset, *args, **kwargs):
    return "Clustering not implemented yet."


class DimensionalityReduction(BaseModel):
    """Class for dimensionality reduction computation

    Args:
        method: Method for dimensionality reduction
        brain_key: Key to store the result in the brain. Only lower case alphanumeric characters and underscores are allowed.
    """

    method: Optional[Literal["umap", "tsne", "pca"]] = Field(
        description="Method to use for dimensionality reduction"
    )
    brain_key: Optional[str] = Field(
        description="Key to store the result in the brain"
    )


def _generate_new_key(dataset, key):
    existing_keys = dataset.list_brain_runs()
    if key not in existing_keys:
        return key
    new_key = key
    i = 1
    while new_key in existing_keys:
        new_key = f"{key}_{i}"
        i += 1
    return new_key


def _can_compute_clustering():
    return False
    # return "@jacobmarks/clustering" in fop.list_enabled_plugins()


def _can_compute_brightness():
    return "@jacobmarks/image_issues" in fop.list_enabled_plugins()


def _can_compute_entropy():
    return "@jacobmarks/image_issues" in fop.list_enabled_plugins()


_clustering_failure_message = "Cannot compute clustering. Make sure the `@jacobmarks/clustering` plugin is installed and enabled."
_brightness_failure_message = "Cannot compute brightness. Make sure the `@jacobmarks/image_issues` plugin is installed and enabled."
_entropy_failure_message = "Cannot compute entropy. Make sure the `@jacobmarks/image_issues` plugin is installed and enabled."
