"""
Query DB.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from .client import get_client, get_ns
from .tables import UserQueryTable


def table(cls):
    # If a project ID was provided, use BigQuery, else in-memory
    project_id = os.environ.get("PROJECT_ID")
    return cls(
        project_id,
        get_client(project_id),
        get_ns(project_id),
        os.environ.get("DATASET_ID"),
    )
