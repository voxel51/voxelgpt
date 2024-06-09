"""
View stage validator

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone as fo

# pylint: disable=relative-beyond-top-level
from .data_inspection_agent import make_data_inspection_tools
from .view_stage_constructor import (
    MatchLabels,
    ToPatches,
    ToEvaluationPatches,
    SelectGroupSlices,
    MatchTags,
    SelectLabels,
)


def _validate_to_patches_stage(view_stage, dataset):
    field_name = view_stage.field
    if field_name not in dataset.get_field_schema():
        raise ValueError(f"Invalid field: {field_name}")
    if dataset.get_field(field_name).document_type != fo.Detections:
        raise ValueError(f"Invalid field type: {field_name}")
    return view_stage


def _validate_to_evaluation_patches_stage(view_stage, dataset):
    dataset_eval_keys = dataset.list_evaluations()
    if len(dataset_eval_keys) == 0:
        raise ValueError("No evaluations found")
    detection_keys = [
        key
        for key in dataset_eval_keys
        if dataset.get_evaluation_info(key).config.type == "detection"
    ]
    if len(detection_keys) == 0:
        raise ValueError("No detection evaluations found")
    eval_key = view_stage.eval_key
    if eval_key not in detection_keys:
        ## Assign the first detection key as a default
        view_stage.eval_key = detection_keys[0]

    return view_stage


def _validate_select_group_slices_stage(view_stage, dataset):
    if dataset.media_type != "group":
        # raise ValueError("Invalid media type")
        #! TO DO: Disambiguate
        view_stage = None

    if view_stage.slices is not None:
        if not all(
            slice_name in dataset.group_slices
            for slice_name in view_stage.slices
        ):
            raise ValueError("Invalid group slices")
    elif view_stage.media_type is not None:
        if view_stage.media_type not in ["image", "video", "3d"]:
            raise ValueError("Invalid media type")

    return view_stage


def _validate_match_tags_stage(view_stage, dataset):
    all_tags = dataset.distinct("tags")
    query_tags = view_stage.tags

    intersection = set(all_tags).intersection(set(query_tags))
    if len(intersection) == 0:
        #! TO DO: Disambiguate
        raise ValueError("No common tags found")

    return view_stage


def _get_label_tags_field(dataset, label_field):
    all_fields = list(dataset.get_field_schema(flat=True).keys())
    for field in all_fields:
        if field.startswith(label_field) and field.endswith("tags"):
            return field
    return None


def _validate_select_labels_stage(view_stage, dataset):
    ## if label field specified, check if tag is in the field
    if view_stage.fields is None:
        return view_stage

    if isinstance(view_stage.fields, str):
        field = view_stage.fields
    else:
        if len(view_stage.fields) != 1:
            return view_stage
        field = view_stage.fields[0]

    tags_subfield = _get_label_tags_field(dataset, field)
    if tags_subfield is None:
        return view_stage

    all_tags = dataset.distinct(tags_subfield)
    query_tags = view_stage.tags
    if isinstance(query_tags, str):
        query_tags = [query_tags]

    intersection = set(all_tags).intersection(set(query_tags))
    if len(intersection) != 0:
        return view_stage
    if len(query_tags) == 1:
        view_stage = MatchLabels(
            fields=field,
            filter_expression='F("label") == "{}"'.format(query_tags[0]),
        )
    return view_stage


def validate_view_stage(view_stage, dataset):
    make_data_inspection_tools(dataset)

    if isinstance(view_stage, ToPatches):
        view_stage = _validate_to_patches_stage(view_stage, dataset)
    elif isinstance(view_stage, ToEvaluationPatches):
        view_stage = _validate_to_evaluation_patches_stage(view_stage, dataset)
    elif isinstance(view_stage, SelectGroupSlices):
        view_stage = _validate_select_group_slices_stage(view_stage, dataset)
    elif isinstance(view_stage, MatchTags):
        view_stage = _validate_match_tags_stage(view_stage, dataset)
    elif isinstance(view_stage, SelectLabels):
        view_stage = _validate_select_labels_stage(view_stage, dataset)

    return view_stage
