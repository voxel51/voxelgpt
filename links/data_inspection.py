"""
Data Inspection Agent

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
from typing import List, Dict, Any

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import fiftyone as fo
from fiftyone import ViewField as F

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_agent_executor_chain, gpt_4o


DATA_INSPECTION_PATH = os.path.join(
    PROMPTS_DIR, "data_inspection_no_view_creation.txt"
)


def _create_data_agent_executor(sample_collection):
    tools = make_data_inspection_tools(sample_collection)
    return _build_agent_executor_chain(gpt_4o, tools, DATA_INSPECTION_PATH)


def run_basic_data_inspection_query(query, sample_collection):
    def data_inspection_func(info):
        query = info["query"]
        response = _create_data_agent_executor(sample_collection).invoke(
            {"input": query}
        )
        return response

    data_inspection_runnable = RunnableLambda(data_inspection_func)
    return data_inspection_runnable.invoke({"query": query})["output"]


def _list_detection_fields(sample_collection):
    return list(
        sample_collection.get_field_schema(
            embedded_doc_type=fo.Detections
        ).keys()
    )


def _list_classification_fields(sample_collection):
    return list(
        sample_collection.get_field_schema(
            embedded_doc_type=fo.Classification
        ).keys()
    )


def _list_polylines_fields(sample_collection):
    return list(
        sample_collection.get_field_schema(
            embedded_doc_type=fo.Polylines
        ).keys()
    )


def _has_geolocation(sample_collection):
    geo_fields = list(
        sample_collection.get_field_schema(
            embedded_doc_type=fo.GeoLocation
        ).keys()
    )
    return len(geo_fields) > 0


def make_data_inspection_tools(sample_collection):
    @tool
    def list_sample_fields() -> Dict[str, str]:
        """Lists the fields in my dataset."""
        return _list_fields(sample_collection)

    @tool
    def list_geolocation_fields() -> List[str]:
        """Lists the geolocation fields in my dataset."""
        return list(
            sample_collection.get_field_schema(
                embedded_doc_type=fo.GeoLocation
            ).keys()
        )

    @tool
    def list_detection_fields() -> List[str]:
        """Lists the detection fields in my dataset."""
        return _list_detection_fields(sample_collection)

    @tool
    def list_classification_fields() -> List[str]:
        """Lists the classification fields in my dataset."""
        return _list_classification_fields(sample_collection)

    @tool
    def list_detection_classes(detection_field: str) -> List[str]:
        """Lists the classes in the specified detection field in my dataset."""
        return sample_collection.distinct(
            F(f"{detection_field}.detections.label")
        )

    @tool
    def list_classification_classes(classification_field: str) -> List[str]:
        """Lists the classes in the specified classification field in my dataset."""
        return sample_collection.distinct(F(f"{classification_field}.label"))

    @tool
    def list_polylines_fields() -> List[str]:
        """Lists the polyline fields in my dataset."""
        return _list_polylines_fields(sample_collection)

    @tool
    def list_polylines_classes(polyline_field: str) -> List[str]:
        """Lists the classes in the specified polyline field in my dataset."""
        return sample_collection.distinct(
            F(f"{polyline_field}.polylines.label")
        )

    @tool
    def list_segmentation_fields() -> List[str]:
        """Lists the segmentation fields in my dataset."""
        return list(
            sample_collection.get_field_schema(
                embedded_doc_type=fo.Segmentation
            ).keys()
        )

    @tool
    def list_keypoints_fields() -> List[str]:
        """Lists the keypoints fields in my dataset."""
        return list(
            sample_collection.get_field_schema(
                embedded_doc_type=fo.Keypoints
            ).keys()
        )

    @tool
    def list_heatmap_fields() -> List[str]:
        """Lists the heatmap fields in my dataset."""
        return list(
            sample_collection.get_field_schema(
                embedded_doc_type=fo.Heatmap
            ).keys()
        )

    @tool
    def get_dataset_name() -> str:
        """Returns the name of the dataset."""
        return sample_collection.name

    @tool
    def get_dataset_length() -> int:
        """Returns the number of samples in the dataset."""
        return sample_collection.count()

    @tool
    def get_dataset_info() -> Dict[str, Any]:
        """Returns the dataset info."""
        return sample_collection.info

    @tool
    def get_dataset_tags() -> List[str]:
        """Returns the tags of the dataset."""
        return sample_collection.tags

    @tool
    def get_dataset_description() -> str:
        """Returns the description of the dataset."""
        return sample_collection.description

    @tool
    def has_metadata() -> bool:
        """Returns whether the dataset has metadata."""
        return (
            sample_collection.exists("metadata").count()
            == sample_collection.count()
        )

    @tool
    def has_geolocation() -> bool:
        """Returns whether the dataset has geolocation data."""
        return _has_geolocation(sample_collection)

    # @tools
    @tool
    def list_brain_runs() -> List[str]:
        """Lists the names of the brain runs in the workspace. This includes
        runs for:
        - computing similarity (e.g., `clip_sim`)
        - computing visualization with dim reduction (e.g., `umap_vis`)
        - computing hardness (e.g., `hardness`)
        - computing mistakennes (e.g., `mistakenness`)
        """
        return sample_collection.list_brain_runs()

    @tool
    def get_brain_run_info(brain_key: str) -> Dict[str, Any]:
        """Returns the info about the brain run specified by `brain_key`. The
        brain run must exist on the dataset."""
        return dict(sample_collection.get_brain_info(brain_key).serialize())

    @tool
    def list_evaluation_runs() -> List[str]:
        """Lists the names of the evaluation runs in the workspace."""
        return sample_collection.list_evaluations()

    @tool
    def get_evaluation_run_info(eval_key: str) -> Dict[str, Any]:
        """Returns the info about the evaluation run specified by `eval_key`. The
        evaluation run must exist on the dataset."""
        return dict(
            sample_collection.get_evaluation_info(eval_key).serialize()
        )

    @tool
    def list_annotation_runs() -> List[str]:
        """Lists the names of the annotation runs in the workspace."""
        return sample_collection.list_annotation_runs()

    @tool
    def get_annotation_run_info(annotation_key: str) -> Dict[str, Any]:
        """Returns the info about the annotation run specified by `annotation_key`.
        The annotation run must exist on the dataset."""
        return dict(
            sample_collection.get_annotation_info(annotation_key).serialize()
        )

    @tool
    def list_custom_runs() -> List[str]:
        """Lists the names of the custom runs in the workspace."""
        return sample_collection.list_runs()

    @tool
    def get_custom_run_info(custom_key: str) -> Dict[str, Any]:
        """Returns the info about the custom run specified by `custom_key`. The
        custom run must exist on the dataset."""
        return dict(sample_collection.get_run_info(custom_key).serialize())

    @tool
    def get_dataset_media_type() -> str:
        """Returns the media type of the dataset. If media type is 'grouped',
        the dataset contains multiple groups slices, each with its own media
        type."""
        return sample_collection.media_type

    @tool
    def get_schema_of_field(field: str) -> Dict[str, Any]:
        """Returns a dictionary containing the schema of the subfields within
        the specified embedded document field."""
        obj = (
            sample_collection.exists(field).select_fields(field).first()[field]
        )
        if isinstance(obj, fo.Detections):
            return obj.detections[0].to_dict().items()
        elif isinstance(obj, fo.Keypoints):
            return obj.keypoints[0].to_dict().items()
        else:
            return obj.to_dict().items()

    @tool
    def get_dataset_group_slices() -> List[str]:
        """Returns the group slices of the dataset. For a dataset with media type
        `group`, this will return the group slices. For a dataset with other
        media types, this will return an empty list.

        """
        res = sample_collection.group_slices
        return res if res is not None else []

    data_inspection_tools = [
        has_metadata,
        has_geolocation,
        list_sample_fields,
        list_geolocation_fields,
        list_classification_fields,
        list_detection_fields,
        list_polylines_fields,
        list_segmentation_fields,
        list_heatmap_fields,
        list_keypoints_fields,
        list_classification_classes,
        list_detection_classes,
        list_polylines_classes,
        list_evaluation_runs,
        list_brain_runs,
        list_annotation_runs,
        list_custom_runs,
        get_dataset_name,
        get_dataset_length,
        get_dataset_info,
        get_dataset_tags,
        get_dataset_description,
        get_dataset_media_type,
        get_evaluation_run_info,
        get_brain_run_info,
        get_annotation_run_info,
        get_custom_run_info,
        get_schema_of_field,
        get_dataset_group_slices,
    ]

    return data_inspection_tools


def _convert_fiftyone_type(fo_type):
    if isinstance(fo_type, fo.StringField):
        return "str"
    elif isinstance(fo_type, fo.IntField):
        return "int"
    elif isinstance(fo_type, fo.FloatField):
        return "float"
    elif isinstance(fo_type, fo.BooleanField):
        return "bool"
    elif isinstance(fo_type, fo.DateTimeField):
        return "datetime"
    else:
        try:
            emb_doc_type = fo_type.document_type_obj

            for lt in [
                fo.Detections,
                fo.Classification,
                fo.Segmentation,
                fo.Heatmap,
                fo.Keypoints,
            ]:
                if issubclass(emb_doc_type, lt):
                    return lt.__name__
        except:
            pass

    return type(fo_type).__name__


def _list_fields(sample_collection):
    return {
        k: _convert_fiftyone_type(v)
        for k, v in dict(sample_collection.get_field_schema()).items()
    }


geo_types = ("GeoNear", "GeoWithin")
all_field_types = (
    "SelectFields",
    "ExcludeFields",
    "FilterField",
    "Exists",
    "Match",
    "GroupBy",
)
label_field_types = (
    "FilterLabels",
    "LimitLabels",
    "SelectLabels",
    "MapLabels",
    "MatchLabels",
    "Match",
)
grouped_field_types = ("SelectGroupSlices",)


def _involves_label_classes(actors, plan):
    for actor, step in zip(actors, plan.steps):
        if actor not in label_field_types:
            continue
        if any([patt in step for patt in ["label", "class", "'", '"']]):
            return True
    return False


def _get_text_sim_runs(dataset):
    text_runs = []
    for run in dataset.list_brain_runs():
        config = dataset.get_brain_info(run).config
        if config.type == "similarity" and config.supports_prompts:
            text_runs.append(run)
    return text_runs


def _get_detection_evaluation_runs(dataset):
    det_eval_runs = []
    for run in dataset.list_evaluations():
        if dataset.get_evaluation_info(run).config.type == "detection":
            det_eval_runs.append(run)
    return det_eval_runs


def _run_default_inspection_for_plan(dataset, actors, plan):
    inspection_results = ""

    all_fields_flag = any([actor in all_field_types for actor in actors])
    geo_types_flag = any([actor in geo_types for actor in actors])
    label_fields_flag = any([actor in label_field_types for actor in actors])
    grouped_flag = any([actor in grouped_field_types for actor in actors])
    match_tags_flag = any([actor in ["MatchTags"] for actor in actors]) or (
        grouped_flag and dataset.media_type != "group"
    )
    label_classes_flag = _involves_label_classes(actors, plan)
    sim_flag = any([actor in ["SortBySimilarity"] for actor in actors])
    patches_flag = any([actor in ["ToPatches"] for actor in actors])
    eval_patches_flag = any(
        [actor in ["ToEvaluationPatches"] for actor in actors]
    )

    ## Basic info
    if all_fields_flag:
        inspection_results += "Dataset has the following fields (type): \n"
        for field, ftype in _list_fields(dataset).items():
            inspection_results += f"- {field} ({ftype})\n"

    ## Geolocation
    if geo_types_flag:
        if _has_geolocation(dataset):
            inspection_results += "Dataset has geolocation data, so you can use `GeoNear` and `GeoWithin` stages.\n"
        else:
            inspection_results += "Dataset does not have geolocation data. You cannot use `GeoNear` and `GeoWithin` stages.\n"

    ## Grouped
    if grouped_flag:
        if dataset.media_type == "group":
            inspection_results += f"Dataset is grouped, so you can use `SelectGroupSlices` stage. The group slices are: {dataset.group_slices}\n"
        else:
            inspection_results += "Dataset is not grouped. You cannot use `SelectGroupSlices` stage. Consider using `MatchTags` or a categorical field.\n"

    ## Classification, Detection, Polylines
    if label_fields_flag:
        cls_fields = _list_classification_fields(dataset)
        if cls_fields:
            inspection_results += f"Dataset has the following classification fields: {cls_fields}\n"
            for cls_field in cls_fields:
                has_confs = (
                    dataset.limit(10).distinct(f"{cls_field}.confidence") != []
                )
                if has_confs:
                    inspection_results += f"Field {cls_field} has confidence values, so it is likely a prediction field.\n"
                else:
                    inspection_results += f"Field {cls_field} does not have confidence values, so it is likely a ground truth field.\n"

                if label_classes_flag:
                    classes = dataset[:1000].distinct(f"{cls_field}.label")
                    if len(classes) < 100:
                        inspection_results += f"Field {cls_field} has the following classes: {classes}\n"
        det_fields = _list_detection_fields(dataset)
        if det_fields:
            inspection_results += (
                f"Dataset has the following detection fields: {det_fields}\n"
            )
            for det_field in det_fields:
                has_confs = (
                    dataset.limit(10).distinct(
                        f"{det_field}.detections.confidence"
                    )
                    != []
                )
                if has_confs:
                    inspection_results += f"Field {det_field} has confidence values, so it is likely a prediction field.\n"
                else:
                    inspection_results += f"Field {det_field} does not have confidence values, so it is likely a ground truth field.\n"

                if label_classes_flag:
                    classes = dataset[:1000].distinct(
                        f"{det_field}.detections.label"
                    )
                    if len(classes) < 100:
                        inspection_results += f"Field {det_field} has the following classes: {classes}\n"
        pl_fields = _list_polylines_fields(dataset)
        if pl_fields:
            inspection_results += (
                f"Dataset has the following polyline fields: {pl_fields}\n"
            )
            for pl_field in pl_fields:
                has_confs = (
                    dataset.limit(10).distinct(
                        f"{pl_field}.polylines.confidence"
                    )
                    != []
                )
                if has_confs:
                    inspection_results += f"Field {pl_field} has confidence values, so it is likely a prediction field.\n"
                else:
                    inspection_results += f"Field {pl_field} does not have confidence values, so it is likely a ground truth field.\n"

                if label_classes_flag:
                    classes = dataset[:1000].distinct(
                        f"{pl_field}.polylines.label"
                    )
                    if len(classes) < 100:
                        inspection_results += f"Field {pl_field} has the following classes: {classes}\n"

        ## Patches
        if patches_flag and not det_fields:
            inspection_results += "Dataset does not have detection fields, so you cannot use `ToPatches` stage.\n"
    else:
        if patches_flag:
            det_fields = _list_detection_fields(dataset)
            if det_fields:
                inspection_results += "Dataset has detection fields, so you can use `ToPatches` stage.\n"
            else:
                inspection_results += "Dataset does not have detection fields, so you cannot use `ToPatches` stage.\n"

    ## Evaluation Patches
    if eval_patches_flag:
        det_eval_runs = _get_detection_evaluation_runs(dataset)
        if not det_eval_runs:
            inspection_results += "Dataset does not have detection evaluation runs, so you cannot use `ToEvaluationPatches` stage.\n"
        else:
            inspection_results += f"Dataset has the following detection evaluation runs: {det_eval_runs}\n"
            for run in det_eval_runs:
                inspection_results += (
                    f"Here is configuration info about the run {run}:\n"
                )
                for k, v in dataset.get_evaluation_info(
                    run
                ).config.__dict__.items():
                    inspection_results += f"    {k}: {v}\n"

    # SortBySimilarity
    if sim_flag:
        text_runs = _get_text_sim_runs(dataset)
        if not text_runs:
            inspection_results += "Dataset does not have text similarity brain runs, so you cannot use `SortBySimilarity` stage. Instead, consider using `MatchTags` or a categorical field.\n"
        if len(text_runs) > 1:
            inspection_results += f"Dataset has multiple text similarity brain runs: {text_runs}\n"
            for run in text_runs:
                inspection_results += (
                    f"Here is configuration info about the run {run}:\n"
                )
                for k, v in dataset.get_brain_info(
                    run
                ).config.__dict__.items():
                    inspection_results += f"    {k}: {v}\n"

    ## Sample Tags
    if match_tags_flag or (sim_flag and not text_runs):
        tags = dataset.distinct("tags")
        inspection_results += f"There is at least one sample with each of the following tags: {tags}\n"

    if inspection_results == "":
        inspection_results = "Given the planned stages, there is no specific information to provide about the dataset.\n"
    return inspection_results
