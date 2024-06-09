"""
View creation classifier.

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


def write_log(log):
    with open("/tmp/log.txt", "a") as f:
        f.write(str(log) + "\n")


def create_view(query, dataset):
    write_log("inside create_view")
    write_log(str(dataset))
    view_creation_plan = create_view_creation_plan(query)
    write_log(str(view_creation_plan))
    view_creation_actors = [
        delegate_view_stage_creation(step) for step in view_creation_plan.steps
    ]
    write_log(str(view_creation_actors))

    inspection_results = None

    view_stages = []
    built_stages = []
    for assignee, step in zip(view_creation_actors, view_creation_plan.steps):
        stage = construct_stage(step, assignee, inspection_results, dataset)
        stage = validate_view_stage(stage, dataset)
        if stage is not None:
            built_stages.append(stage.build())

    _reorder_built_stages_if_needed(built_stages)
    for stage in built_stages:
        view_stages.append(stage)

    write_log(str(view_stages))

    view = dataset
    for stage in view_stages:
        view = view.add_stage(stage)

    return view


def _reorder_built_stages_if_needed(built_stages):
    ## Put all GeoNear and GeoWithin stages at the beginning
    for i, stage in enumerate(built_stages):
        if isinstance(stage, (fo.GeoNear, fo.GeoWithin)):
            built_stages.insert(0, built_stages.pop(i))
