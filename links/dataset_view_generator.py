import re

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

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
    template=MISTAKENNESS_FIELD_PROMPT_TEMPLATE
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

def generate_evaluation_prompt(dataset, eval_key):
    field_names = dataset.first().field_names

    prompt = EVALUATION_PROMPT_TEMPLATE.format(eval_key=eval_key)

    if f"{eval_key}_tp" in field_names:
        prompt += EVAL_FIELDS_PROMPT_TEMPLATE.format(eval_tp_field=f"{eval_key}_tp", eval_fp_field=f"{eval_key}_fp", eval_fn_field=f"{eval_key}_fn")

    return prompt


def generate_mistakenness_prompt(dataset, brain_key):
    field_names = dataset.first().field_names

    brc = dataset.get_brain_info(brain_key).config
    mistakenness_field = brc.mistakenness_field
    prompt = mistakenness_field_prompt.format(mistakenness_field=mistakenness_field)

    missing_field = brc.missing_field
    if missing_field in field_names:
        prompt += missing_field_prompt.format(missing_field=missing_field)

    spurious_field = brc.spurious_field
    if spurious_field in field_names:
        prompt += spurious_field_prompt.format(spurious_field=spurious_field)

    return prompt

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

def generate_runs_prompt(dataset, runs):
    ## If there are no runs, return an empty string
    if len(runs) == 0:
        return ""

    header = "Here is the relevant information about the runs that were run on this dataset:\n"
    prompt = header

    if "uniqueness" in runs:
        uniqueness_field = runs["uniqueness"]
        uniqueness_prompt = UNIQUENESS_PROMPT.format(uniqueness_field=uniqueness_field)
        prompt += uniqueness_prompt

    if "hardness" in runs:
        hardness_field = runs["hardness"]["hardness_field"]
        label_field = runs["hardness"]["label_field"]
        hardness_prompt = HARDNESS_PROMPT.format(hardness_field=hardness_field, label_field=label_field)
        prompt += hardness_prompt

    if "image_similarity" in runs:
        image_similarity_key = runs["image_similarity"]
        image_similarity_prompt = IMAGE_SIMILARITY_PROMPT.format(image_similarity_key=image_similarity_key)
        prompt += image_similarity_prompt

    if "text_similarity" in runs:
        text_similarity_key = runs["text_similarity"]
        text_similarity_prompt = TEXT_SIMILARITY_PROMPT.format(text_similarity_key=text_similarity_key, brain_key=text_similarity_key["key"])
        prompt += text_similarity_prompt

    if "mistakenness" in runs:
        mistakenness_prompt = generate_mistakenness_prompt(dataset, runs["mistakenness"]["key"])
        prompt += mistakenness_prompt

    if "evaluation" in runs:
        evaluation_prompt = generate_evaluation_prompt(dataset, runs["evaluation"]["key"])
        prompt += evaluation_prompt

    if "metadata" in runs:
        prompt += "You can also use the `metadata` key to access the metadata for each sample.\n"

    return prompt

def load_dataset_view_prompt_prefix_template():
    with open("prompts/dataset_view_generator_prefix.txt", "r") as f:
        prefix = f.read()
    return prefix

def generate_dataset_view_prompt_prefix(available_fields, label_classes):
    template = load_dataset_view_prompt_prefix_template()
    prompt = PromptTemplate(
        input_variables=["available_fields", "label_classes"],
        template=template
    )

    return prompt.format(available_fields=available_fields, label_classes=label_classes)

def generate_dataset_view_prompt(
        dataset,
        required_runs,
        available_fields,
        label_classes,
        view_stage_descriptions,
        examples_prompt
    ):

    prompt = generate_dataset_view_prompt_prefix(available_fields, label_classes)
    prompt += generate_runs_prompt(dataset, required_runs)
    prompt += view_stage_descriptions
    prompt += examples_prompt
    return prompt

def generate_dataset_view_text(
        dataset,
        required_runs,
        available_fields,
        label_classes,
        view_stage_descriptions,
        examples_prompt
    ):
    prompt = generate_dataset_view_prompt(
        dataset,
        required_runs,
        available_fields,
        label_classes,
        view_stage_descriptions,
        examples_prompt
    )

    response = llm.call_as_llm(prompt)
    return response.strip()

def remove_whitespace(stage_str):
    return re.sub(
        r'\s+', lambda m: ' ' if len(m.group(0)) == 1 else '',
        stage_str
        )

def split_into_stages(stages_text):
    with open("view_stages_list.txt", "r") as f:
        view_stages = f.read().splitlines()
    pattern = ','+'|,'.join(view_stages)[:-1]

    st = stages_text[1:-1].replace(', ', ',').replace('\n', '')
    st = st.replace('\r', '').replace('\'', "\"")
    x = re.finditer(pattern, st)

    stages = []
    spans = []
    for match in x:
        spans.append(match.span())

    spans = spans[::-1]
    for i, span in enumerate(spans):
        if i == 0:
            stages.append(st[span[0]+1:])
        else:
            stages.append(st[span[0]+1:spans[i-1][0]])
    if len(stages) != 0:
        stages.append(st[:spans[-1][0]])
    else:
        stages.append(st)

    stages = stages[::-1]
    stages = [remove_whitespace(stage) for stage in stages]
    return stages

def get_gpt_view_stage_strings(
        dataset,
        required_brain_runs,
        available_fields,
        label_classes,
        view_stage_descriptions,
        examples_prompt
    ):
    response = generate_dataset_view_text(
        dataset,
        required_brain_runs,
        available_fields,
        label_classes,
        view_stage_descriptions,
        examples_prompt
    ).strip()

    response = response.replace("LEFTBRACKET", "{")
    response = response.replace("RIGHTBRACKET", "}")

    if '_MORE_' in response:
        return '_MORE_'
    elif "_CONFUSED_" in response:
        return "_CONFUSED_"
    else:
        return split_into_stages(response)
