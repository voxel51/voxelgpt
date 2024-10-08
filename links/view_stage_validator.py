"""
View stage validator

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import re

import fiftyone as fo
from fiftyone import ViewField as F

# pylint: disable=relative-beyond-top-level
from .utils import (
    _gt_field_names,
    _pred_field_names,
    tp_field_names,
    fp_field_names,
    fn_field_names,
)
from .data_inspection import (
    _get_classification_evaluation_runs,
    _get_detection_evaluation_runs,
    _get_text_sim_runs,
    _list_detection_fields,
    _list_classification_fields,
    _list_polylines_fields,
)
from .view_stage_constructor import (
    Match,
    MatchLabels,
    FilterLabels,
    ToPatches,
    ToEvaluationPatches,
    SelectGroupSlices,
    MatchTags,
    SelectLabels,
    SortBySimilarity,
    SelectFields,
    ExcludeFields,
    Exists,
)


def _validate_to_patches_stage(view_stage, dataset):
    det_fields = _list_detection_fields(dataset)
    if len(det_fields) == 0:
        return "No detection fields found"

    field_name = view_stage.field
    if field_name in det_fields:  ## valid
        return view_stage

    if field_name in _gt_field_names and any(
        field in det_fields for field in _gt_field_names
    ):
        view_stage.field = next(
            field for field in _gt_field_names if field in det_fields
        )
    elif field_name in _pred_field_names and any(
        field in det_fields for field in _pred_field_names
    ):
        view_stage.field = next(
            field for field in _pred_field_names if field in det_fields
        )
    else:
        view_stage.field = det_fields[
            0
        ]  ## Assign the first detection field as a default

    return view_stage


def _validate_to_evaluation_patches_stage(view_stage, dataset):
    eval_keys = _get_detection_evaluation_runs(dataset)
    if len(eval_keys) == 0:
        return "No detection evaluations found"

    eval_key = view_stage.eval_key
    if eval_key in eval_keys:
        return view_stage  ## valid

    view_stage.eval_key = eval_keys[
        0
    ]  ## Assign the first detection evaluation as a default

    return view_stage


def _validate_select_group_slices_stage(view_stage, dataset):
    if dataset.media_type != "group":
        return "No: Dataset is not a group dataset"

    if view_stage.slices is not None:
        if not all(
            slice_name in dataset.group_slices
            for slice_name in view_stage.slices
        ):
            return "No: Invalid group slices"
    elif view_stage.media_type is not None:
        if view_stage.media_type not in ["image", "video", "3d"]:
            return "No: Invalid media type"

    return view_stage


def _validate_match_tags_stage(view_stage, dataset):
    all_tags = dataset.distinct("tags")
    query_tags = view_stage.tags

    intersection = set(all_tags).intersection(set(query_tags))
    if len(intersection) == 0:
        #! TO DO: Disambiguate
        return "No: No common tags found"

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
    elif len(query_tags) == 1:
        view_stage = MatchLabels(
            fields=field,
            filter_expression='F("label") == "{}"'.format(query_tags[0]),
        )
        return view_stage
    else:
        return "No: No common tags found"


def _validate_sort_by_similarity_stage(view_stage, dataset):
    sim_keys = _get_text_sim_runs(dataset)
    if len(sim_keys) == 0:
        return "No: No text similarity runs found"

    sim_key = view_stage.brain_key
    if sim_key in sim_keys:
        return view_stage  ## valid

    view_stage.brain_key = sim_keys[
        0
    ]  ## Assign the first text similarity run as a default

    return view_stage


def _convert_contains_to_equals(expression):
    # Regular expression pattern to match the .contains("...") part
    pattern = r'\.contains\("([^"]+)"\)'

    # Use re.sub to replace the matched pattern with the desired result
    result = re.sub(pattern, r' == "\1"', expression)
    return result


def _extract_label_class(expression):
    if "==" in expression or "!=" in expression:
        # Regular expression pattern to capture the values in double quotes after == or !=
        pattern = r'==\s*"([^"]+)"|!=\s*"([^"]+)"'

        # Find all matches in the expression
        matches = re.findall(pattern, expression)

        # Extract the captured values from the matches
        values = [match[0] or match[1] for match in matches]
        return values
    elif "is_in" in expression:
        # Regular expression pattern to capture the values in double quotes after is_in
        pattern = r"is_in\(\[([^\]]+)\]\)"
        _match = re.search(pattern, expression)
        if _match:
            items_str = _match.group(1)
            # Split the items by comma and strip quotes and spaces
            items = [item.strip().strip('"') for item in items_str.split(",")]
            return items
    else:
        return []


def _get_class_names(dataset, field_name):
    try:
        ftype = dataset.get_field(field_name).document_type
    except:
        return []
    if ftype == fo.Detections:
        return dataset.distinct(f"{field_name}.detections.label")
    elif ftype == fo.Classification:
        return dataset.distinct(f"{field_name}.label")
    elif ftype == fo.Polylines:
        return dataset.distinct(f"{field_name}.polylines.label")
    else:
        return []


det_eval_patts = ("fp", "fn", "tp")
cls_eval_patts = ("true", "false")


def _set_field(view_stage, field):
    if hasattr(view_stage, "field"):
        view_stage.field = field
    else:
        view_stage.fields = [field]
    return view_stage


def _get_field(view_stage):
    return (
        view_stage.field
        if hasattr(view_stage, "field")
        else view_stage.fields[0]
    )


def _has_predictions(dataset, field):
    view = dataset.filter_labels(field, F("confidence"))
    return view.count() > 0


def _resolve_label_field(view_stage, dataset, gt=True, doc_type=None):
    if isinstance(view_stage, str):
        return None
    ## gt=True --> resolve ground truth field
    ## gt=False --> resolve prediction field

    def _resolve_label_field_evaluations():

        filter_expr = view_stage.filter_expression.lower()
        if "eval" in filter_expr:
            if any(patt in filter_expr for patt in det_eval_patts):
                eval_keys = _get_detection_evaluation_runs(dataset)
                if len(eval_keys) != 0:
                    eval_key = eval_keys[0]
                    eval_run = dataset.get_evaluation_info(eval_key)
                    return (
                        eval_run.config.gt_field
                        if gt
                        else eval_run.config.pred_field
                    )
            if any(patt in filter_expr for patt in cls_eval_patts):
                eval_keys = _get_classification_evaluation_runs(dataset)
                if len(eval_keys) != 0:
                    eval_key = eval_keys[0]
                    eval_run = dataset.get_evaluation_info(eval_key)
                    return (
                        eval_run.config.gt_field
                        if gt
                        else eval_run.config.pred_field
                    )

        return None

    try:
        field = _resolve_label_field_evaluations()
    except:
        field = None

    if field is not None and field != None:
        return field

    candidate_fields = (
        _list_detection_fields(dataset)
        + _list_classification_fields(dataset)
        + _list_polylines_fields(dataset)
    )

    if len(candidate_fields) != 0:
        for field in candidate_fields:
            try:
                hp = _has_predictions(dataset, field)
                if hp and not gt:
                    return field
                elif not hp and gt:
                    return field
            except Exception as e:
                continue

    ## if no candidate fields found, try to resolve otherwise
    if gt:
        for field_name in _gt_field_names:
            if dataset.has_field(field_name):
                return field_name
    else:
        pred_field = _get_field(view_stage)
        if dataset.has_field(pred_field):
            return pred_field

        if doc_type is not None and doc_type == "classification":
            candidate_fields = _list_classification_fields(dataset)
        elif doc_type is not None and doc_type == "detection":
            candidate_fields = _list_detection_fields(dataset)
        else:
            candidate_fields = (
                _list_classification_fields(dataset)
                + _list_detection_fields(dataset)
                + _list_polylines_fields(dataset)
            )

        for field in candidate_fields:
            if _has_predictions(dataset, field):
                return field

    return None


def _resolve_ground_truth_field(view_stage, dataset):
    return _resolve_label_field(view_stage, dataset, gt=True)


def _resolve_prediction_field(view_stage, dataset, doc_type=None):
    return _resolve_label_field(
        view_stage, dataset, gt=False, doc_type=doc_type
    )


def _validate_filter_labels_fields(view_stage, dataset):
    field = _get_field(view_stage)

    # ignore multi-field case for now
    if isinstance(field, list) and len(field) > 1:
        return view_stage
    elif isinstance(field, list):
        field = field[0]

    if dataset.has_field(field) and issubclass(
        dataset.get_field(field).document_type, fo.Label
    ):
        return view_stage

    if (
        field.endswith("predictions")
        and field != "predictions"
        and not dataset.has_field(field)
    ):
        field = field.replace("predictions", "")
        if field.endswith("_") or field.endswith(" "):
            field = field[:-1]
        if dataset.has_field(field) and issubclass(
            dataset.get_field(field).document_type, fo.Label
        ):
            return _set_field(view_stage, field)

    for dataset_field in list(dataset.get_field_schema().keys()):
        if dataset_field.lower().replace("_", "_") == field.lower().replace(
            "_", "_"
        ):
            if issubclass(
                dataset.get_field(dataset_field).document_type, fo.Label
            ):
                return _set_field(view_stage, dataset_field)

    if field == "predictions":
        field = _resolve_prediction_field(view_stage, dataset)
        if field is not None:
            return _set_field(view_stage, field)

    if field == "ground_truth":
        field = _resolve_ground_truth_field(view_stage, dataset)
        if field is not None:
            return _set_field(view_stage, field)

    return view_stage


def _validate_filter_labels_classes(view_stage, dataset):
    if isinstance(view_stage, str):
        return view_stage

    filter_expr = view_stage.filter_expression

    ## handle label classes
    filter_expr = _convert_contains_to_equals(filter_expr)
    view_stage.filter_expression = filter_expr
    if "label" not in filter_expr:
        view_stage.filter_expression = filter_expr
        return view_stage

    label_class_names = _extract_label_class(filter_expr)
    has_matching_class_name = False

    if len(label_class_names) > 0:
        field = _get_field(view_stage)
        field = field if isinstance(field, str) else field[0]
        all_class_names = _get_class_names(dataset, field)
        all_class_names_lower = {
            name.lower(): name for name in all_class_names
        }
        if len(all_class_names) == 0:
            return view_stage

        for class_name in label_class_names:
            if class_name in all_class_names:
                has_matching_class_name = True
            elif class_name not in all_class_names and (
                class_name.lower() in all_class_names_lower
            ):
                new_class_name = all_class_names_lower[class_name.lower()]
                filter_expression = filter_expression.replace(
                    f"{class_name}", f"{new_class_name}"
                )
                has_matching_class_name = True

    ## if no matching class name, check for text similarity runs
    if not has_matching_class_name:
        text_sim_runs = _get_text_sim_runs(dataset)
        if len(text_sim_runs) > 0:
            view_stage = SortBySimilarity(
                text=f"A photo of a {label_class_names[0]}"
            )
            return view_stage

    view_stage.filter_expression = filter_expr

    ## Replace "A.label" with "label" in the filter expression
    pattern = r"\b\w+\.\w+\.(\w+)\b|\b\w+\.(\w+)\b"

    # Function to perform the replacement
    def _replace_label(expr):
        if expr.group(1):
            return expr.group(1)
        if expr.group(2):
            return expr.group(2)

    # Use re.sub to replace the matched pattern with the desired result
    filter_expr = re.sub(pattern, _replace_label, filter_expr)

    ## Handle "==" and "!=" explicitly:
    if ").label" in filter_expr:
        eq_pattern = r'==\s*"([^"]+)"'
        if re.search(eq_pattern, filter_expr):
            name = re.search(eq_pattern, filter_expr).group(1)
            filter_expr = 'F("label") == "{}"'.format(name)

        neq_pattern = r'!=\s*"([^"]+)"'
        if re.search(neq_pattern, filter_expr):
            name = re.search(neq_pattern, filter_expr).group(1)
            filter_expr = 'F("label") != "{}"'.format(name)

    view_stage.filter_expression = filter_expr
    return view_stage


def _evaluate_classifications_manual(gt_field, pred_field, eval_type):
    ##!TODO: Add confidence and other aspects
    eq_patt = "==" if eval_type == "True" else "!="
    return Match(
        filter_expression=f"F('{gt_field}.label') {eq_patt} F('{pred_field}.label')"
    )


def _validate_filter_labels_eval_cls(
    view_stage, dataset, gt_field, pred_field
):
    filter_expr = view_stage.filter_expression

    eval_type = None

    ## Handle TP, FP, FN explicitly:
    for tp_patt in tp_field_names:
        if tp_patt in filter_expr:
            filter_expr = filter_expr.replace(tp_patt, "True")
            eval_type = "True"
    for false_patt in fp_field_names + fn_field_names:
        if false_patt in filter_expr:
            filter_expr = filter_expr.replace(false_patt, "fp")
            eval_type = "False"

    if eval_type is None:
        return view_stage

    ## get evaluation keys
    eval_keys = _get_classification_evaluation_runs(dataset)
    if "eval" in filter_expr and "eval" not in eval_keys:
        if len(eval_keys) == 0:
            return _evaluate_classifications_manual(
                gt_field, pred_field, eval_type
            )
        new_key = eval_keys[0]
        filter_expr = filter_expr.replace("eval", new_key)

        ## Replace "False" --> False, "True" --> True
        filter_expr = filter_expr.replace("'False'", "False").replace(
            '"False"', "False"
        )
        filter_expr = filter_expr.replace("'True'", "True").replace(
            '"True"', "True"
        )

    view_stage.filter_expression = filter_expr
    return view_stage


def _validate_filter_labels_eval_det(view_stage, dataset):
    filter_expr = view_stage.filter_expression

    ## Handle TP, FP, FN explicitly:
    for tp_patt in tp_field_names:
        if tp_patt in filter_expr:
            filter_expr = filter_expr.replace(tp_patt, "tp")
    for fp_patt in fp_field_names:
        if fp_patt in filter_expr:
            filter_expr = filter_expr.replace(fp_patt, "fp")
    for fn_patt in fn_field_names:
        if fn_patt in filter_expr:
            filter_expr = filter_expr.replace(fn_patt, "fn")

    view_stage.filter_expression = filter_expr

    eval_keys = _get_detection_evaluation_runs(dataset)
    if "eval" in filter_expr and "eval" not in eval_keys:
        if len(eval_keys) == 0:
            return "No: no detection evaluations found"
        new_key = eval_keys[0]
        filter_expr = filter_expr.replace("eval", new_key)

    view_stage.filter_expression = filter_expr
    return view_stage


def _validate_filter_labels_evaluations(view_stage, dataset):
    if isinstance(view_stage, str):
        return view_stage

    filter_expr = view_stage.filter_expression
    if "eval" not in filter_expr:
        return view_stage

    gt_field = _resolve_ground_truth_field(view_stage, dataset)
    gt_field_type = (
        "classification"
        if gt_field in _list_classification_fields(dataset)
        else (
            "detection"
            if gt_field in _list_detection_fields(dataset)
            else None
        )
    )

    pred_field = _resolve_prediction_field(
        view_stage, dataset, doc_type=gt_field_type
    )

    if gt_field is None or pred_field is None or gt_field == pred_field:
        return view_stage

    ### classification evaluation
    if gt_field in _list_classification_fields(
        dataset
    ) and pred_field in _list_classification_fields(dataset):
        return _validate_filter_labels_eval_cls(
            view_stage, dataset, gt_field, pred_field
        )

    ## detection evaluation
    if gt_field in _list_detection_fields(
        dataset
    ) and pred_field in _list_detection_fields(dataset):
        return _validate_filter_labels_eval_det(view_stage, dataset)

    ##! TO DO Polylines, segmentations, etc.
    return view_stage


def _validate_filter_labels_stage(view_stage, dataset):
    ## handle field names
    view_stage = _validate_filter_labels_fields(view_stage, dataset)
    ## handle evaluations
    view_stage = _validate_filter_labels_evaluations(view_stage, dataset)
    ## handle label classes
    view_stage = _validate_filter_labels_classes(view_stage, dataset)
    return view_stage


def _validate_fields_stage(view_stage, dataset):
    validated_fields = []

    def _validate_field(field):
        if dataset.has_field(field):
            validated_fields.append(field)
        elif field == "detections":
            det_fields = _list_detection_fields(dataset)
            if len(det_fields) > 0:
                validated_fields.append(det_fields[0])

    fields = view_stage.fields
    if isinstance(fields, str):
        fields = [fields]

    for field in fields:
        _validate_field(field)

    view_stage.fields = validated_fields
    return view_stage


def _validate_exists_stage(view_stage, dataset):
    field = view_stage.field
    if dataset.has_field(field):
        return view_stage

    elif field == "detections":
        det_fields = _list_detection_fields(dataset)
        if len(det_fields) > 0:
            view_stage.field = det_fields[0]
            return view_stage
    elif field == "classification":
        classif_fields = _list_classification_fields(dataset)
        if len(classif_fields) > 0:
            view_stage.field = classif_fields[0]
            return view_stage
    elif field == "polylines":
        polyline_fields = _list_polylines_fields(dataset)
        if len(polyline_fields) > 0:
            view_stage.field = polyline_fields[0]
            return view_stage
    return "No: Field not found"


def validate_view_stage(view_stage, dataset):
    try:
        if isinstance(view_stage, ToPatches):
            view_stage = _validate_to_patches_stage(view_stage, dataset)
        elif isinstance(view_stage, ToEvaluationPatches):
            view_stage = _validate_to_evaluation_patches_stage(
                view_stage, dataset
            )
        elif isinstance(view_stage, SelectGroupSlices):
            view_stage = _validate_select_group_slices_stage(
                view_stage, dataset
            )
        elif isinstance(view_stage, MatchTags):
            view_stage = _validate_match_tags_stage(view_stage, dataset)
        elif isinstance(view_stage, SelectLabels):
            view_stage = _validate_select_labels_stage(view_stage, dataset)
        elif isinstance(view_stage, SortBySimilarity):
            view_stage = _validate_sort_by_similarity_stage(
                view_stage, dataset
            )
        elif isinstance(view_stage, (MatchLabels, FilterLabels)):
            view_stage = _validate_filter_labels_stage(view_stage, dataset)
        elif isinstance(view_stage, (SelectFields, ExcludeFields)):
            view_stage = _validate_fields_stage(view_stage, dataset)
        elif isinstance(view_stage, Exists):
            view_stage = _validate_exists_stage(view_stage, dataset)
    except Exception as e:
        return view_stage
    return view_stage
