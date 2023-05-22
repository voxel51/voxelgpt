"""
Query validator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

# pylint: disable=relative-beyond-top-level
from .utils import get_moderator


def moderate_query(query):
    return get_moderator().run(query)

