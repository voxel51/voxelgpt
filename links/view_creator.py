"""
View creator.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone as fo

# pylint: disable=relative-beyond-top-level
from .view_creation_planner import create_view_creation_plan
from .view_stage_delegator import delegate_view_stage_creation
from .view_stage_constructor import construct_stage
from .view_stage_validator import validate_view_stage
from .utils import has_metadata


def write_log(log):
    with open("/tmp/log.txt", "a") as f:
        f.write(str(log) + "\n")


def create_view_from_plan(sample_collection, view_creation_plan):
    impossible_stages = []
    view_creation_actors = []

    for step in view_creation_plan.steps:
        if step.lower().startswith("no"):
            impossible_stages.append(step)
        else:
            view_creation_actors.append(delegate_view_stage_creation(step))

    view_stages = []
    stage_reprs = []
    built_stages = []
    for assignee, step in zip(view_creation_actors, view_creation_plan.steps):
        stage = construct_stage(step, assignee, sample_collection)
        stage = validate_view_stage(stage, sample_collection)
        write_log(str(stage))
        if stage is not None:
            if isinstance(stage, str):
                impossible_stages.append(step + " - " + stage)
            else:
                built_stages.append(stage.build())
                stage_reprs.append(str(stage.__repr__()))

    _compute_metadata_if_needed(sample_collection, stage_reprs)
    _reorder_built_stages_if_needed(built_stages)
    for stage in built_stages:
        view_stages.append(stage)

    view = sample_collection
    try:
        for stage in view_stages:
            view = view.add_stage(stage)
    except Exception as e:
        return None, None

    return view, stage_reprs


def _reorder_built_stages_if_needed(built_stages):
    ## Put all GeoNear and GeoWithin stages at the beginning
    for i, stage in enumerate(built_stages):
        if isinstance(stage, (fo.GeoNear, fo.GeoWithin)):
            built_stages.insert(0, built_stages.pop(i))

    ## Put GeoNear at the very beginning
    for i, stage in enumerate(built_stages):
        if isinstance(stage, fo.GeoNear):
            built_stages.insert(0, built_stages.pop(i))


def _compute_metadata_if_needed(sample_collection, stage_reprs):
    if "metadata" in "".join(stage_reprs) and not has_metadata(
        sample_collection
    ):
        sample_collection.compute_metadata()
