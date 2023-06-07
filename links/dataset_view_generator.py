"""
Dataset view generator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import re

from langchain.prompts import PromptTemplate

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_cache
from .tag_selector import select_tags


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINKS_DIR = os.path.join(ROOT_DIR, "links")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

VIEW_STAGES_LIST_PATH = os.path.join(ROOT_DIR, "view_stages_list.txt")
VIEW_GENERATOR_PREFIX_PATH = os.path.join(
    PROMPTS_DIR, "dataset_view_generator_prefix.txt"
)

UNIQUENESS_PROMPT_TEMPLATE = """
A uniqueness run determines how unique each image is in the dataset. Its results are stored in the {uniqueness_field} field on the samples.
When converting a natural language query into a DatasetView, if you determine that the uniqueness of the images is important, a view stage should use the {uniqueness_field} field.
"""

HARDNESS_PROMPT_TEMPLATE = """
A hardness run scores each image based on how difficult it is to classify for a specified label field. In this task, the hardness of each sample for the {label_field} field is has been scored, and its results are stored in the {hardness_field} field on the samples.
"""

IMAGE_SIMILARITY_PROMPT_TEMPLATE = """
An image_similarity run determines determines how similar each image is to another image. You can use the {image_similarity_key} key to access the results of this run and sort images by similarity.
"""

TEXT_SIMILARITY_PROMPT_TEMPLATE = """
A text_similarity run determines determines how similar each image is to a user-specified input text prompt. You can use the {text_similarity_key} key to access the results of this run and find images that most resemble the description in the user-input text prompt. You can use these and only these brian_key values brain_key="{brain_key}" for an output using sort_by_similarity.
"""

MISTAKENNESS_FIELD_PROMPT_TEMPLATE = """
A mistakenness run determines how mistaken each image is in the dataset. Its results are stored in the {mistakenness_field} field on the samples.
When converting a natural language query into a DatasetView, if you determine that the mistakenness of the images is important, the following fields store relevant information:
- {mistakenness_field}: the mistakenness score for each image
"""

mistakenness_field_prompt = PromptTemplate(
    input_variables=["mistakenness_field"],
    template=MISTAKENNESS_FIELD_PROMPT_TEMPLATE,
)

missing_field_prompt = PromptTemplate(
    input_variables=["missing_field"],
    template="- {missing_field}: the missing score for each image\n",
)

spurious_field_prompt = PromptTemplate(
    input_variables=["spurious_field"],
    template="- {spurious_field}: the spurious score for each image\n",
)

EVALUATION_PROMPT_TEMPLATE = """
An evaluation run computes metrics, statistics, and reports assessing the accuracy of model predictions for classifications, detections, and segmentations. You can use the {eval_key} key to access the results of this run, including TP, FP, and FNs.
"""

EVAL_FIELDS_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["eval_tp_field", "eval_fp_field", "eval_fn_field"],
    template="""- {eval_tp_field}: the true positive score for each image
- {eval_fp_field}: the false positive score for each image
- {eval_fn_field}: the false negative score for each image
""",
)


UNIQUENESS_PROMPT = PromptTemplate(
    input_variables=["uniqueness_field"],
    template=UNIQUENESS_PROMPT_TEMPLATE,
)

HARDNESS_PROMPT = PromptTemplate(
    input_variables=["hardness_field", "label_field"],
    template=HARDNESS_PROMPT_TEMPLATE,
)

IMAGE_SIMILARITY_PROMPT = PromptTemplate(
    input_variables=["image_similarity_key"],
    template=IMAGE_SIMILARITY_PROMPT_TEMPLATE,
)

TEXT_SIMILARITY_PROMPT = PromptTemplate(
    input_variables=["text_similarity_key", "brain_key"],
    template=TEXT_SIMILARITY_PROMPT_TEMPLATE,
)

SIMILARITY_QUERY_PROMPT_PATH = os.path.join(
    PROMPTS_DIR, "similarity_query_extractor_prompt.txt"
)

DETECTION_KEYWORDS = (
    "_fp",
    "_fn",
    "_tp",
    "FP",
    "FN",
    "TP",
)

CLASSIFICATION_KEYWORDS = ("False", "True")

TEXT_SIM_KEYWORDS = (
    "show",
    "display",
    "find me",
    "images",
    "pictures",
    "photos",
    "videos",
    "samples",
)


def generate_evaluation_prompt(sample_collection, eval_key):
    schema = sample_collection.get_field_schema()

    prompt = EVALUATION_PROMPT_TEMPLATE.format(eval_key=eval_key)

    if f"{eval_key}_tp" in schema:
        prompt += EVAL_FIELDS_PROMPT_TEMPLATE.format(
            eval_tp_field=f"{eval_key}_tp",
            eval_fp_field=f"{eval_key}_fp",
            eval_fn_field=f"{eval_key}_fn",
        )

    return prompt


def generate_mistakenness_prompt(sample_collection, brain_key):
    schema = sample_collection.get_field_schema()

    brc = sample_collection.get_brain_info(brain_key).config
    mistakenness_field = brc.mistakenness_field
    prompt = mistakenness_field_prompt.format(
        mistakenness_field=mistakenness_field
    )

    missing_field = brc.missing_field
    if missing_field in schema:
        prompt += missing_field_prompt.format(missing_field=missing_field)

    spurious_field = brc.spurious_field
    if spurious_field in schema:
        prompt += spurious_field_prompt.format(spurious_field=spurious_field)

    return prompt


def generate_runs_prompt(sample_collection, runs):
    ## If there are no runs, return an empty string
    if len(runs) == 0:
        return ""

    header = "Here is the relevant information about the runs that were run on this dataset:\n"
    prompt = header

    if "uniqueness" in runs:
        uniqueness_field = runs["uniqueness"]
        uniqueness_prompt = UNIQUENESS_PROMPT.format(
            uniqueness_field=uniqueness_field
        )
        prompt += uniqueness_prompt

    if "hardness" in runs:
        hardness_field = runs["hardness"]["hardness_field"]
        label_field = runs["hardness"]["label_field"]
        hardness_prompt = HARDNESS_PROMPT.format(
            hardness_field=hardness_field, label_field=label_field
        )
        prompt += hardness_prompt

    if "image_similarity" in runs:
        image_similarity_key = runs["image_similarity"]
        image_similarity_prompt = IMAGE_SIMILARITY_PROMPT.format(
            image_similarity_key=image_similarity_key
        )
        prompt += image_similarity_prompt

    if "text_similarity" in runs:
        text_similarity_key = runs["text_similarity"]
        text_similarity_prompt = TEXT_SIMILARITY_PROMPT.format(
            text_similarity_key=text_similarity_key,
            brain_key=text_similarity_key["key"],
        )
        prompt += text_similarity_prompt

    if "mistakenness" in runs:
        mistakenness_prompt = generate_mistakenness_prompt(
            sample_collection, runs["mistakenness"]["key"]
        )
        prompt += mistakenness_prompt

    if "evaluation" in runs:
        evaluation_prompt = generate_evaluation_prompt(
            sample_collection, runs["evaluation"]["key"]
        )
        prompt += evaluation_prompt

    if "metadata" in runs:
        prompt += "You can also use the `metadata` key to access the metadata for each sample.\n"

    return prompt


def load_dataset_view_prompt_prefix_template():
    cache = get_cache()
    key = "dataset_view_prompt_prefix"
    if key not in cache:
        with open(VIEW_GENERATOR_PREFIX_PATH, "r") as f:
            cache[key] = f.read()
    return cache[key]


def generate_dataset_view_prompt_prefix(available_fields, label_classes):
    template = load_dataset_view_prompt_prefix_template()
    prompt = PromptTemplate(
        input_variables=["available_fields", "label_classes"],
        template=template,
    )

    return prompt.format(
        available_fields=available_fields, label_classes=label_classes
    )


def generate_dataset_view_prompt(
    sample_collection,
    required_runs,
    available_fields,
    label_classes,
    view_stage_descriptions,
    examples_prompt,
):

    prompt = generate_dataset_view_prompt_prefix(
        available_fields, label_classes
    )
    prompt += generate_runs_prompt(sample_collection, required_runs)
    prompt += view_stage_descriptions
    prompt += examples_prompt
    return prompt


def generate_dataset_view_text(
    sample_collection,
    required_runs,
    available_fields,
    label_classes,
    view_stage_descriptions,
    examples_prompt,
):
    prompt = generate_dataset_view_prompt(
        sample_collection,
        required_runs,
        available_fields,
        label_classes,
        view_stage_descriptions,
        examples_prompt,
    )
    response = get_llm().call_as_llm(prompt)
    return response.strip()


def remove_whitespace(stage_str):
    return re.sub(
        r"\s+", lambda m: " " if len(m.group(0)) == 1 else "", stage_str
    )


def split_into_stages(stages_text):
    with open(VIEW_STAGES_LIST_PATH, "r") as f:
        view_stages = f.read().splitlines()

    upper_to_lower = {
        stage.upper().replace("_", ""): stage for stage in view_stages
    }

    if stages_text[0] == "[" and stages_text[-1] == "]":
        st = stages_text[1:-1]
    elif stages_text[0] == "_" and stages_text[-1] == "_":
        st = stages_text[1:-1]
    else:
        st = stages_text
    st = st.replace(", ", ",").replace("\n", "")
    st = st.replace("\r", "").replace("'", '"')
    if st[:8] == "dataset.":
        st = st[8:]

    pattern = "," + "|,".join(view_stages)[:-1]
    x = re.finditer(pattern, st)

    stages = []
    spans = []
    for match in x:
        spans.append(match.span())

    spans = spans[::-1]
    for i, span in enumerate(spans):
        if i == 0:
            stages.append(st[span[0] + 1 :])
        else:
            stages.append(st[span[0] + 1 : spans[i - 1][0]])
    if len(stages) != 0:
        stages.append(st[: spans[-1][0]])
    else:
        stages.append(st)

    stages = stages[::-1]
    stages = [remove_whitespace(stage) for stage in stages]

    for i, stage in enumerate(stages):
        stage_name = stage.split("(")[0]
        stage_name_compressed = stage_name.replace("_", "")
        if stage_name not in view_stages:
            if stage_name_compressed in upper_to_lower:
                num_chars = len(stage_name)
                stages[i] = (
                    upper_to_lower[stage_name_compressed] + stage[num_chars:]
                )

    return stages


def _get_first_image_text_similarity_key(sample_collection):
    for run in sample_collection.list_brain_runs():
        info = sample_collection.get_brain_info(run)
        if (
            "Similarity" in info.config.cls
            and info.config.supports_prompts
            and not info.config.patches_field
        ):
            return run
    return None


def _convert_matches_to_text_similarities(
    stage, sample_collection, required_brain_runs, unmatched_classes
):
    """
    if model picks a non-existent class and you have text similarity run,
    convert the match to a text similarity stage
    """

    if "text_similarity" not in required_brain_runs:
        text_sim_key = _get_first_image_text_similarity_key(sample_collection)
        if not text_sim_key:
            return stage
    else:
        text_sim_key = required_brain_runs["text_similarity"]["key"]

    def _replace_stage(entity):
        new_stage = "".join(
            [
                f"sort_by_similarity('{entity}'",
                f", brain_key = '{text_sim_key}', k = 100)",
            ]
        )
        return new_stage

    def _insensitive_match(entity, stage):
        el = entity.lower()
        st = stage.lower()
        if el in st or el[:-1] in st:
            return True

    def _loop_over_unmatched_classes(stage):
        if "sort_by_similarity" in stage:
            return stage
        for entity in unmatched_classes:
            if _insensitive_match(entity, stage):
                return _replace_stage(entity)
        return stage

    if "match" in stage and "label" in stage:
        return _loop_over_unmatched_classes(stage)
    elif "filter_labels" in stage:
        return _loop_over_unmatched_classes(stage)
    else:
        return stage


def _validate_match(
    stage,
):
    """
    if model predicts `match(F(field.detections.label) == "class_name")`, then
    fix it by subbing in `contains()`.
    if model has `label` in both sides of filter statement, then fix it by
    removing the first.
    """

    def _correct_length_stage(stage):
        if "filter" not in stage:
            if "F" not in stage:
                return stage
            contents_F = stage.split("F(")[1].split(")")[0]
            if "detections" in contents_F:
                return stage
            else:
                field_name = contents_F.split(".")[0]
                det_subfield_name = field_name[:-1] + '.detections"'
                return stage.replace(contents_F, det_subfield_name)
        else:
            contents = stage[6:-1]
            split_contents = contents.split("length()")
            length_expr = split_contents[-1]
            start = split_contents[0]

            contents_F = stage.split("F(")[1].split(")")[0]
            F_expr = f"F({contents_F})".replace(".label", "")

            if "==" or "!=" in start:
                negation = "~" in contents
                eq_expr = "==" if "==" in start else "!="
                class_name = start.split(eq_expr)[1].split(")")[0].strip()

                filter_expr = f'.filter(F("label") {eq_expr} {class_name})'
                contents = f"{F_expr}{filter_expr}.length(){length_expr}"
                if negation:
                    contents = f"~({contents})"
                return f"match({contents})"

            return stage

    if "length" in stage:
        return _correct_length_stage(stage)
    elif "detections.label" not in stage:
        new_stage = stage
    elif "contains" in stage or "is_subset" in stage:
        new_stage = stage
    elif "is_in" in stage:
        new_stage = stage.replace("is_in", "contains")
    elif "filter" in stage:
        a, b = stage.split("filter")
        if "label" in a and "label" in b:
            a = a.replace(".label", "")
            new_stage = a + "filter" + b
        else:
            new_stage = stage
    else:
        field_name = stage.split("F")[1].split(".")[0].strip()
        if "==" in stage:
            class_name = stage.split("==")[1].strip()[:-1]
        else:
            class_name = stage.split("!=")[1].strip()[:-1]
        new_stage = (
            f'match(F{field_name}.detections.label").contains({class_name}))'
        )

    return new_stage


def _get_unique_label_classes_dict(label_classes):
    unique_classes_dict = {}
    for class_maps in label_classes.values():
        for class_map in class_maps:
            if type(class_map) == str:
                continue
            for ner_class, label_class in class_map.items():
                if ner_class not in unique_classes_dict:
                    unique_classes_dict[ner_class] = label_class

    return unique_classes_dict


def _validate_stages_ner(stage, label_classes):
    """
    Ensure that class names were subbed in correctly.
    """

    if "sort_by_similarity" in stage or "map_labels" in stage:
        return stage
    else:
        new_stage = stage
        unique_label_classes_dict = _get_unique_label_classes_dict(
            label_classes
        )
        for ner_class, label_class in unique_label_classes_dict.items():
            if ner_class in stage and type(label_class) == str:
                new_stage = new_stage.replace(ner_class, label_class)
        return new_stage


def get_unique_class_list(label_classes):
    unique_classes = []
    for class_list in label_classes.values():
        for class_name in class_list:
            if class_name not in unique_classes:
                unique_classes.append(class_name)
    return unique_classes


def _validate_label_class_case(stage, label_classes):
    """
    Ensure that class names have correct case.
    """

    unique_classes = get_unique_class_list(label_classes)

    if "sort_by_similarity" in stage or "map_labels" in stage:
        return stage
    else:
        new_stage = stage
        for class_name in unique_classes:
            new_stage = re.sub(
                class_name, class_name, new_stage, flags=re.IGNORECASE
            )
        return new_stage


def _infer_stage_evaluation_type(stage):
    if any(keyword in stage for keyword in DETECTION_KEYWORDS):
        return "detection"
    elif any(keyword in stage for keyword in CLASSIFICATION_KEYWORDS):
        return "classification"
    else:
        return None


def _get_first_detection_eval_key(sample_collection):
    eval_keys = sample_collection.list_evaluations()
    for ek in eval_keys:
        eval_cls = sample_collection.get_evaluation_info(ek).config.cls
        if "openimages" in eval_cls:
            return ek
        elif "coco" in eval_cls:
            return ek
        elif "activitynet" in eval_cls:
            return ek
    return None


def _get_first_classification_eval_key(sample_collection):
    eval_keys = sample_collection.list_evaluations()
    for ek in eval_keys:
        eval_cls = sample_collection.get_evaluation_info(ek).config.cls
        if "classification" in eval_cls:
            return ek
    return None


def _get_first_valid_eval_key(sample_collection, eval_type):
    if eval_type == "detection":
        return _get_first_detection_eval_key(sample_collection)
    elif eval_type == "classification":
        return _get_first_classification_eval_key(sample_collection)
    else:
        return None


def _correct_eval_run(stage, sample_collection, runs):
    if "evaluation" in runs:
        eval_key = runs["evaluation"]["key"]
    else:
        eval_type = _infer_stage_evaluation_type(stage)
        if not eval_type:
            return "_MORE_"

        eval_key = _get_first_valid_eval_key(sample_collection, eval_type)

    if eval_key:
        return stage.replace("EVAL_KEY", eval_key)
    else:
        return "_MORE_"


def _correct_uniqueness_run(stage, sample_collection, runs):
    uniqueness_field = None
    if "uniqueness" in runs:
        uniqueness_field = runs["uniqueness"]["uniqueness_field"]
    else:
        brain_runs = sample_collection.list_brain_runs()
        brain_runs = [
            sample_collection.get_brain_info(br) for br in brain_runs
        ]

        for br in brain_runs:
            if br.config.method == "uniqueness":
                uniqueness_field = br.config.uniqueness_field
                break

    if uniqueness_field:
        return stage.replace("UNIQUENESS_FIELD", uniqueness_field)
    else:
        return "_MORE_"


def _correct_text_sim_run(stage, sample_collection, runs):
    text_sim_key = None
    if "text_similarity" in runs:
        text_sim_key = runs["text_similarity"]["key"]
    else:
        brain_runs = sample_collection.list_brain_runs()
        brain_runs = [
            sample_collection.get_brain_info(br) for br in brain_runs
        ]

        for br in brain_runs:
            if "Similarity" in br.config.cls and br.config.supports_prompts:
                text_sim_key = br.config.key
                break

    if text_sim_key:
        return stage.replace("TEXT_SIM_KEY", text_sim_key)
    else:
        return "_MORE_"


def _correct_image_sim_run(stage, sample_collection, runs):
    image_sim_key = None
    if "image_similarity" in runs:
        image_sim_key = runs["image_similarity"]["key"]
    else:
        brain_runs = sample_collection.list_brain_runs()
        brain_runs = [
            sample_collection.get_brain_info(br) for br in brain_runs
        ]

        for br in brain_runs:
            if "Similarity" in br.config.cls:
                image_sim_key = br.config.key
                break

    if image_sim_key:
        return stage.replace("IMAGE_SIM_KEY", image_sim_key)
    else:
        return "_MORE_"


def _validate_runs(stage, sample_collection, required_brain_runs):
    if "EVAL_KEY" in stage:
        new_stage = _correct_eval_run(
            stage,
            sample_collection,
            required_brain_runs,
        )
    elif "UNIQUENESS_FIELD" in stage:
        new_stage = _correct_uniqueness_run(
            stage,
            sample_collection,
            required_brain_runs,
        )
    elif "TEXT_SIM_KEY" in stage:
        new_stage = _correct_text_sim_run(
            stage,
            sample_collection,
            required_brain_runs,
        )
    elif "IMAGE_SIM_KEY" in stage:
        new_stage = _correct_image_sim_run(
            stage,
            sample_collection,
            required_brain_runs,
        )
    else:
        new_stage = stage

    return new_stage


def _has_confidences(sample_collection, field):
    _, path = sample_collection._get_label_field_path(field, "confidence")

    # Expensive
    # return sample_collection.bounds(path) != (None, None)

    # Faster
    confs = sample_collection.exists(field).limit(1).values(path, unwind=True)
    return confs and confs[0] is not None


def _validate_label_fields(
    stage, sample_collection, label_classes, required_brain_runs
):
    """
    Ensure that label fields are correct.
    """
    if "map_labels" in stage:
        return stage

    label_fields = list(label_classes.keys())

    def _get_ground_truth_field():
        return label_fields[0]

    def _get_predictions_field():
        if len(label_fields) == 1:
            return label_fields[0]

        for field in label_fields:
            try:
                if _has_confidences(sample_collection, field):
                    return field
            except:
                pass

        if len(label_fields) > 0:
            return label_fields[0]
        else:
            if "evaluation" in required_brain_runs:
                return required_brain_runs["evaluation"]["pred_field"]
            return None

    new_stage = stage

    if "ground_truth" in stage and "ground_truth" not in label_fields:
        if "confidence" in stage:
            field = _get_predictions_field()
        else:
            field = _get_ground_truth_field()

        new_stage = new_stage.replace("ground_truth", field)

    if '"gt' in stage and '"gt' not in label_fields:
        if "confidence" in stage:
            field = _get_predictions_field()
        else:
            field = _get_ground_truth_field()

        new_stage = new_stage.replace('"gt', f'"{field}')

    if "predictions" in stage and "predictions" not in label_fields:
        pred_field = _get_predictions_field()
        if pred_field:
            new_stage = new_stage.replace("predictions", pred_field)

    return new_stage


def _split_match_tags_contents(contents):
    if "[" in contents:
        tags_arg = contents.split("[")[1].split("]")[0]
        tags = tags_arg.split(",")
        other = "".join(contents.split("]")[1])
    else:
        tags = [contents.split(",")[0]]
        other = "," + ",".join(contents.split(",")[1:])
        if set(other).issubset(set(", ")):
            other = ""

    tags = [t.strip()[1:-1] for t in tags]
    return tags, other


def _validate_match_tags(stage, sample_collection):
    """
    Ensure that match_tags expression is sensible.
    """
    collection_tags = sample_collection.distinct("tags")

    contents = stage[len("match_tags(") : -1]
    tags, other_args = _split_match_tags_contents(contents)
    selected_tags = select_tags(tags, collection_tags)
    if len(selected_tags) == 0:
        return "_MORE_"

    if len(selected_tags) == 1:
        return f'match_tags("{selected_tags[0]}"{other_args})'

    selected_tags = [f'"{t}"' for t in selected_tags]
    return f'match_tags([{",".join(selected_tags)}]{other_args})'


def _validate_sort_by(stage, sample_collection, required_brain_runs):
    contents = stage[8:-1]

    if "." in contents:
        return stage
    elif "F" in contents:
        F_expr = contents.split(",")[0]
        if F_expr[-2:] != '")':
            return stage

    num_commas = contents.count(",")

    if num_commas > 1:
        return stage
    elif num_commas == 0:
        field = contents
    else:
        field, order = contents.split(",")

    field = field.replace('"', "").replace("'", "")
    if "F(" in field:
        field = field.split("F(")[1].split(")")[0]

    if field in sample_collection.first().field_names:
        return stage
    else:
        if "text_similarity" not in required_brain_runs:
            sim_key = _get_first_image_text_similarity_key(sample_collection)
            if not sim_key:
                return "_MORE_"
        else:
            sim_key = required_brain_runs["text_similarity"]["key"]

    return f'sort_by_similarity("{field}", brain_key="{sim_key}", k = 100)'


def _remove_match_labels_field_name(stage):
    contents = stage[13:-1]
    if "," in contents:
        F_expr, fields_expr = contents.split(",")
        if "F(" in fields_expr:
            F_expr, fields_expr = fields_expr, F_expr
    else:
        F_expr = contents

    F_contents = F_expr.split("(")[1].split(")")[0]
    if "." not in F_contents:
        return stage

    field = F_contents.split(".")[0].replace('"', "").replace("'", "")
    stage = stage.replace(F_contents, '"label"')

    if "," in contents:
        filter_arg, fields_arg = contents.split(",")
        if "F(" in fields_arg:
            filter_arg, fields_arg = fields_arg, filter_arg
        filter_arg = filter_arg.replace(field, "label")
        contents = f"{filter_arg}, {fields_arg}"
    else:
        contents = contents.replace(field, "label")
    stage = f"match_labels({contents})"

    if "fields" not in stage:
        contents = stage[13:-1]
        stage = f'match_labels({contents}, fields="{field}")'
        return stage
    else:
        return stage


def _remove_match_labels_contains(stage):
    if "contains" not in stage:
        return stage

    return stage.replace("contains", "is_in")


def _replace_match_labels_label(stage, label_classes):
    contents = stage[13:-1]
    if "," in contents:
        F_expr, fields_expr = contents.split(",")
        if "F(" in fields_expr:
            F_expr, fields_expr = fields_expr, F_expr
    else:
        F_expr = contents
    F_contents = F_expr.split("(")[1].split(")")[0]
    for field_name in label_classes.keys():
        if field_name in F_contents:
            stage = stage.replace(field_name, "label")
            if "fields" not in stage:
                contents = stage[13:-1]
                stage = f'match_labels({contents}, fields="{field_name}")'
            return stage
    return stage


def _postprocess_match_labels(stage, label_classes):
    stage = _remove_match_labels_field_name(stage)
    stage = _remove_match_labels_contains(stage)
    stage = _replace_match_labels_label(stage, label_classes)
    return stage


def _validate_match_labels(stage, label_classes):
    """
    Correct a few common errors in match_labels stage.
    """

    def get_label_field(contents, used_classes):

        ### first check for label field names in the stage string
        for field_name in label_classes.keys():
            if field_name in contents:
                return field_name

        ### if no label field names are found, check for class name matches
        for fn, classes in label_classes.items():
            if len(set(classes).union(used_classes)) > 0:
                return fn

        ### if no label field names or class names are found, return None
        return None

    def _convert_count_match_labels_to_match(stage):
        contents = stage[13:-1]
        class_name = contents.split("count(")[1].split(")")[0]
        class_name = class_name.replace('"', "").replace("'", "")
        field = get_label_field(contents, [class_name])
        length_expr = contents.split("count(")[1].split(")")[1].split(",")[0]
        filter_expr = f'F("label") == "{class_name}"'
        return f'match(F("{field}.detections").filter({filter_expr}).length(){length_expr})'

    def _convert_length_match_labels_to_match(stage):
        contents = stage[13:-1]
        length_expr = contents.split("length()")[1].split(",")[0]
        if "filter=" in contents:
            filter_expr = contents.split("filter=")[1].split(".length()")[0]
        elif "filter" in contents:
            filter_expr = contents.split("filter")[1].split(".length()")[0]
        else:
            filter_expr = contents.split(".length()")[0]

        unique_classes = get_unique_class_list(label_classes)
        present_classes = [
            class_name
            for class_name in unique_classes
            if class_name in contents
        ]
        field = get_label_field(contents, present_classes)

        if len(present_classes) == 1 and (
            "!=" in filter_expr or "~" in filter_expr
        ):
            filter_expr = f'F("label") != "{present_classes[0]}"'
        elif len(present_classes) == 1 and (
            "==" in filter_expr
            or "contains" in filter_expr
            or "in" in filter_expr
        ):
            filter_expr = f'F("label") == "{present_classes[0]}"'
        elif len(present_classes) > 1 and "!=" or "~" in filter_expr:
            filter_expr = f'~ (F("label").is_in({present_classes}))'
        else:
            filter_expr = f'F("label").is_in({present_classes})'

        if field is None:
            return "_MORE_"

        return f'match(F("{field}.detections").filter({filter_expr}).length(){length_expr})'

    stage = stage.replace("in_classes", "is_in")
    stage = stage.replace("contains_labels", "contains")

    if "length()" in stage:
        return _convert_length_match_labels_to_match(stage)

    if "count" in stage:
        return _convert_count_match_labels_to_match(stage)

    contents = stage[13:-1]
    if "labels" in contents and "{" in contents:
        unique_classes = get_unique_class_list(label_classes)
        present_classes = [
            class_name
            for class_name in unique_classes
            if class_name in contents
        ]
        field_names = [fn for fn in label_classes.keys() if fn in contents]

        if len(field_names) == 0:
            field_names_str = ""
        elif len(field_names) == 1:
            field_names_str = f', fields = "{field_names[0]}"'
        else:
            field_strs = [f'"{field_name}"' for field_name in field_names]
            field_names_str = f", fields = {field_strs}"

        if len(present_classes) == 1:
            classes_str = f'F("label") == "{present_classes[0]}"'
        else:
            class_strs = [f'"{class_name}"' for class_name in present_classes]
            classes_str = f'F("label").is_in({class_strs})'

        contents = f"filter = {classes_str}{field_names_str}"
        if ".label" in contents:
            contents = contents.replace(".label", "")
        return f"match_labels({contents})"
    elif "is_in" in contents:
        is_in = contents.split("is_in([")[1].split("])")[0]
        elems = [elem.replace('"', "") for elem in is_in.split(",")]
        label_field = get_label_field(contents, elems)
        field_names_str = f', fields = "{label_field}"'
        class_strs = [f"{class_name}" for class_name in elems]
        classes_str = f'F("label").is_in({class_strs})'
        contents = f"filter = {classes_str}{field_names_str}"
        if ".label" in contents:
            contents = contents.replace(".label", "")
        return f"match_labels({contents})"
    elif "filter=" in contents:
        if ".label" in contents:
            contents = contents.replace(".label", "")
        for field in label_classes.keys():
            if f'F("{field}")' in contents:
                contents = contents.replace(f'F("{field}")', f'F("label")')

                if "fields" not in contents:
                    contents = f'{contents}, fields = "{field}"'

                return f"match_labels({contents})"
            elif f'F("{field}.confidence")' in contents:
                contents = contents.replace(
                    f'F("{field}.confidence")', f'F("confidence")'
                )
                if "fields" not in contents:
                    contents = f'{contents}, fields = "{field}"'

                return f"match_labels({contents})"
        return stage
    elif "==" in contents and "label" not in contents:
        unique_classes = get_unique_class_list(label_classes)
        present_classes = [
            class_name
            for class_name in unique_classes
            if class_name in contents
        ]

        if len(present_classes) == 1:
            present_class = present_classes[0]
            field = get_label_field(contents, present_classes)
            if not field:
                return stage
            contents = (
                f"fields = '{field}', filter = F('label') == '{present_class}'"
            )
            return f"match_labels({contents})"
        else:
            return stage
    elif not any(
        [
            patt in contents
            for patt in ["F", "label", "filter", "id", "bool", "tags"]
        ]
    ):
        unique_classes = get_unique_class_list(label_classes)
        present_classes = [
            class_name
            for class_name in unique_classes
            if class_name in contents
        ]

        if len(present_classes) == 1:
            present_class = present_classes[0]
            fields = [fn for fn in label_classes.keys() if fn in contents]
            if len(fields) == 0:
                field_str = ""
            if len(fields) == 1:
                field_str = f", fields = '{fields[0]}'"
            else:
                field_str = f", fields = {fields}"

            filter_str = f"filter = F('label') == '{present_class}'"
            contents = f"{filter_str}{field_str}"
            return f"match_labels({contents})"
        else:
            return stage

    else:
        return stage


def _add_parens(expression, split_char):
    parts = expression.split(split_char)
    parts = [p.strip() for p in parts]
    parts = [f"({p})" if p[0] != "(" else p for p in parts]
    return f" {split_char} ".join(parts)


def _validate_filter_labels(stage, label_classes):
    """
    Correct a few common errors in filter_labels stage.

    """
    contents = stage[14:-1]
    args = contents.split(",")

    ##### correct label_field if needed
    if args[0].strip() == "None":
        for field in label_classes.keys():
            if field in contents:
                contents = contents.replace("None", f'"{field}"')
                break
    elif len(args) == 1:
        for field in label_classes.keys():
            if field in contents:
                contents = contents.replace(field, "label")
                contents = f'"{field}", {contents}'
                return f"filter_labels({contents})"
        tmp = args[0].split('"')
        label_field = tmp[1]
    else:
        label_field = args[0][1:-1]
        if label_field not in label_classes.keys():
            for field in label_classes.keys():
                if field in label_field and field != label_field:
                    contents = contents.replace(args[0], f'"{field}"')
                    break

    ##### correct three-argument hallucination of form 'filter_labels("label_field", "==", "class_name")'
    eq_pattern = r'"([^"]+)",\s*"==",\s*"([^"]+)"'
    eq_matches = re.findall(eq_pattern, contents)
    if eq_matches:
        match = eq_matches[0]
        label_field = match[0]
        class_name = match[1]
        return f'filter_labels("{label_field}", F("label") == "{class_name}")'

    ##### correct three-argument hallucination of form 'filter_labels("label_field", "!=", "class_name")'
    neq_pattern = r'"([^"]+)",\s*"!=",\s*"([^"]+)"'
    neq_matches = re.findall(neq_pattern, contents)
    if neq_matches:
        match = neq_matches[0]
        label_field = match[0]
        class_name = match[2]
        return f'filter_labels("{label_field}", F("label") != "{class_name}")'

    ##### correct second argument if needed
    if len(args) == 2:
        arg1 = args[1].strip()
        if arg1[0] == '"' and arg1[-1] == '"':
            contents = contents.replace(arg1, f'F("label") == {arg1}')
    if ").label" in contents:
        contents = re.sub(r"F\(\"(.+?)\"\)\.label", r'F("label")', contents)
    if ".label" in contents:
        contents = re.sub(r'F\("([^"]*?)\.label"', r'F("label"', contents)
    if ".confidence" in contents:
        contents = re.sub(
            r'F\("([^"]*?)\.confidence"', r'F("confidence"', contents
        )
    if '"label"' in contents and 'F("label")' not in contents:
        contents = contents.replace('"label"', 'F("label")')
    if '"confidence"' in contents and 'F("confidence")' not in contents:
        contents = contents.replace('"confidence"', 'F("confidence")')

    ##### correct AND statements if needed
    if " and " in contents:
        contents = contents.replace(" and ", " & ")

    ##### correct OR statements if needed
    if " or " in contents:
        contents = contents.replace(" or ", " | ")

    ##### add parens if needed
    args = contents.split(",")
    if len(args) == 2:
        arg0, arg1 = args
    else:
        arg0 = args[0]
        arg1 = ", ".join(args[1:])

    if "&" in contents:
        arg1 = _add_parens(arg1, "&")
    if "|" in contents:
        arg1 = _add_parens(arg1, "|")

    contents = f"{arg0}, {arg1}"
    return f"filter_labels({contents})"


def _extract_query_from_examples(examples_prompt):
    example_lines = examples_prompt.split("\n")
    query = example_lines[-2]
    if query.startswith("Input: "):
        query = query[7:]
    return query


def attempt_to_disambiguate(
    response,
    sample_collection,
    required_brain_runs,
    label_classes,
    unmatched_classes,
    examples_prompt,
):

    ### Case 1: one label field
    if len(label_classes) == 1:
        label_field = list(label_classes.keys())[0]
        class_list = label_classes[label_field]

        if len(class_list) == 1:
            ### Case 1a: one class
            return f'[filter_labels(F("{label_field}") == "{class_list[0]}")]'
        elif len(class_list) > 1:
            ### Case 1b: multiple classes
            return f'[filter_labels(F("{label_field}").is_in({class_list}))]'

    ### If we don't have text similarity, we can't do anything else
    if "text_similarity" not in required_brain_runs:
        sim_key = _get_first_image_text_similarity_key(sample_collection)
        if not sim_key:
            return response
    else:
        sim_key = required_brain_runs["text_similarity"]["key"]

    if len(unmatched_classes) == 1:
        ### Case 2: 1 unmatched class & text_similarity
        uc = unmatched_classes[0]
        return f'[sort_by_similarity("{uc}", brain_key="{sim_key}", k=100)]'
    else:
        ### Case 3: multiple unmatched classes & text_similarity
        query = _extract_query_from_examples(examples_prompt)
        return f'[sort_by_similarity("{query}", brain_key="{sim_key}", k=100)]'


def _validate_negation_operator(stage):
    if "!F" in stage:
        stage = stage.replace("!F", "~F")
    return stage


def _get_false_patterns(stage):
    false_patterns = [
        r",\s*False",
        r",\s*invert\s*=\s*True",
    ]

    if stage.startswith("match_labels"):
        return false_patterns
    elif stage.startswith("match_tags"):
        return false_patterns
    elif stage.startswith("exists"):
        return false_patterns
    else:
        return false_patterns + [r",\s*bool\s*=\s*False"]


def _validate_bool_condition(stage):
    false_patterns = _get_false_patterns(stage)

    for pattern in false_patterns:
        false_matches = re.findall(pattern, stage)
        if false_matches:
            stage = re.sub(pattern, "", stage)
            opening_paren_index = stage.index("(")
            # Extract the function name
            stage_name = stage[:opening_paren_index]

            # Extract the contents
            contents = stage[opening_paren_index + 1 : -1]
            return f"{stage_name}(~({contents}))"
    return stage


def load_similarity_query_prompt():
    cache = get_cache()
    key = "similarity_query_prefix"
    if key not in cache:
        with open(SIMILARITY_QUERY_PROMPT_PATH, "r") as f:
            cache[key] = f.read()
    return cache[key]


def extract_similarity_query(stage):
    pattern = r'sort_by_similarity\("([^"]+)"'
    query = re.search(pattern, stage).group(1)
    sim_query_prompt = load_similarity_query_prompt().replace("QUERY", query)
    new_query = get_llm().call_as_llm(sim_query_prompt).strip()
    return stage.replace(query, new_query)


def _validate_text_similarity(stage):
    if "sort_by_similarity" not in stage:
        return stage
    if any(keyword in stage for keyword in TEXT_SIM_KEYWORDS):
        return extract_similarity_query(stage)
    else:
        return stage


def _get_sort_by_similarity_stage(stages):
    for stage in stages:
        if "sort_by_similarity(" in stage:
            return stage
    return None


def _handle_duplicate_stages(stages):
    sim_sort_stage = _get_sort_by_similarity_stage(stages)
    if not sim_sort_stage:
        return stages

    sim_query = sim_sort_stage.split("(")[1].split(",")[0]
    sim_query = sim_query.replace('"', "").replace("'", "")

    verified_stages = []
    for stage in stages:
        if "sort_by_similarity" in stage:
            verified_stages.append(stage)
        elif sim_query not in stage:
            verified_stages.append(stage)
    return verified_stages


def _get_label_type(label_field, sample_collection):
    return (
        sample_collection.exists(label_field)
        .first()[label_field]
        .__class__.__name__
    )


def _validate_eval_result(stage, sample_collection, required_brain_runs):
    if "match_labels" not in stage or "eval" not in stage:
        return stage
    elif "evaluation" not in required_brain_runs:
        return stage

    pred_field = required_brain_runs["evaluation"]["pred_field"]
    eval_key = required_brain_runs["evaluation"]["key"]

    label_type = _get_label_type(pred_field, sample_collection)
    if label_type in ["Detections", "Polylines"]:
        for patt in ["fp", "tp", "fn"]:
            if f"{eval_key}_{patt}" in stage:
                return f'match_labels(filter=F("{eval_key}") == "{patt}", fields="{pred_field}")'

    return stage


def _postprocess_stages(
    stages,
    sample_collection,
    required_brain_runs,
    label_classes,
    unmatched_classes,
):
    new_stages = []

    for stage in stages:
        _stage = stage
        _stage = _convert_matches_to_text_similarities(
            _stage, sample_collection, required_brain_runs, unmatched_classes
        )
        _stage = _validate_label_fields(
            _stage, sample_collection, label_classes, required_brain_runs
        )
        _stage = _validate_label_class_case(_stage, label_classes)
        _stage = _validate_stages_ner(_stage, label_classes)
        _stage = _validate_runs(_stage, sample_collection, required_brain_runs)
        if "match(" in stage:
            _stage = _validate_match(_stage)
        if "filter_labels" in _stage:
            _stage = _validate_filter_labels(_stage, label_classes)
        if "match_labels" in _stage:
            _stage = _validate_match_labels(_stage, label_classes)
            _stage = _postprocess_match_labels(_stage, label_classes)
        if "match_tags" in _stage:
            _stage = _validate_match_tags(_stage, sample_collection)
        if "sort_by(" in _stage:
            _stage = _validate_sort_by(
                _stage, sample_collection, required_brain_runs
            )
        _stage = _validate_negation_operator(_stage)
        _stage = _validate_bool_condition(_stage)
        _stage = _validate_text_similarity(_stage)
        _stage = _validate_eval_result(
            _stage, sample_collection, required_brain_runs
        )
        new_stages.append(_stage)

    new_stages = _handle_duplicate_stages(new_stages)

    return new_stages


def get_gpt_view_stage_strings(
    sample_collection,
    required_brain_runs,
    available_fields,
    label_classes,
    unmatched_classes,
    view_stage_descriptions,
    examples_prompt,
):
    response = generate_dataset_view_text(
        sample_collection,
        required_brain_runs,
        available_fields,
        label_classes,
        view_stage_descriptions,
        examples_prompt,
    ).strip()

    response = response.replace("LEFTBRACKET", "{")
    response = response.replace("RIGHTBRACKET", "}")

    if "more" in response.lower() or "confused" in response.lower():
        response = attempt_to_disambiguate(
            response,
            sample_collection,
            required_brain_runs,
            label_classes,
            unmatched_classes,
            examples_prompt,
        )

    if "more" in response.lower():
        return "_MORE_"

    if "confused" in response.lower():
        return "_CONFUSED_"

    stages = split_into_stages(response)
    return _postprocess_stages(
        stages,
        sample_collection,
        required_brain_runs,
        label_classes,
        unmatched_classes,
    )
