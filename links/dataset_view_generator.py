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
from .utils import get_llm


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
    with open(VIEW_GENERATOR_PREFIX_PATH, "r") as f:
        return f.read()


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

    pattern = "," + "|,".join(view_stages)[:-1]

    st = stages_text[1:-1].replace(", ", ",").replace("\n", "")
    st = st.replace("\r", "").replace("'", '"')
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
        if stage_name not in view_stages and stage_name in upper_to_lower:
            num_chars = len(stage_name)
            stages[i]= upper_to_lower[stage_name] + stage[num_chars:]

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
    stages,
    sample_collection,
    required_brain_runs,
    unmatched_classes
):
    '''
    if model picks a non-existent class and you have text similarity run,
    convert the match to a text similarity stage
    '''
    if 'text_similarity' not in required_brain_runs:
        text_sim_key = _get_first_image_text_similarity_key(sample_collection)
        if not text_sim_key:
            return stages
    else:
        text_sim_key = required_brain_runs['text_similarity']['key']
    
    def _replace_stage(entity):
        new_stage = ''.join(
            [
                f"sort_by_similarity('{entity}'",
                f", brain_key = '{text_sim_key}', k = 100)"
            ])
        return new_stage
    
    def _loop_over_unmatched_classes(stage):
        if 'sort_by_similarity' in stage:
            return stage
        for entity in unmatched_classes:
            if entity in stage:
                return _replace_stage(entity)
        return stage
        
    verified_stages = []
    for stage in stages:
        if 'match' in stage and 'label' in stage:
            verified_stages.append(_loop_over_unmatched_classes(stage))
        elif 'filter_labels' in stage:
            verified_stages.append(_loop_over_unmatched_classes(stage))
        else:
            verified_stages.append(stage)
    return verified_stages

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

    if "_MORE_" in response:
        return "_MORE_"
    elif "_CONFUSED_" in response:
        return "_CONFUSED_"
    else:
        stages = split_into_stages(response)
        stages = _convert_matches_to_text_similarities(
            stages,
            sample_collection,
            required_brain_runs,
            unmatched_classes
        )
        return stages
