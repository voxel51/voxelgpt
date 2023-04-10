from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
llm = OpenAI(temperature=0.0)

UNIQUENESS_PROMPT_TEMPLATE = """
A uniqueness brain run determines how unique each image is in the dataset. Its results are stored in the {uniqueness_field} field on the samples.
When converting a natural language query into a DatasetView, if you determine that the uniqueness of the images is important, a view stage should use the {uniqueness_field} field.
"""

HARDNESS_PROMPT_TEMPLATE = """
A hardness brain run scores each image based on how difficult it is to classify for a specified label field. In this task, the hardness of each sample for the {label_field} field is has been scored, and its results are stored in the {hardness_field} field on the samples.
"""

IMAGE_SIMILARITY_PROMPT_TEMPLATE = """
An image_similarity brain run determines determines how similar each image is to another image. You can use the {image_similarity_key} brain key to access the results of this brain run and sort images by similarity.
"""

TEXT_SIMILARITY_PROMPT_TEMPLATE = """
A text_similarity brain run determines determines how similar each image is to a user-specified input text prompt. You can use the {text_similarity_key} brain key to access the results of this brain run and find images that most resemble the description in the user-input text prompt.
"""

MISTAKENNESS_FIELD_PROMPT_TEMPLATE = """
A mistakenness brain run determines how mistaken each image is in the dataset. Its results are stored in the {mistakenness_field} field on the samples.
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

mistakenness_eval_prompt = PromptTemplate(
    input_variables=["eval_tp_field", "eval_fp_field", "eval_fn_field"],
    template="""- {eval_tp_field}: the true positive score for each image
- {eval_fp_field}: the false positive score for each image
- {eval_fn_field}: the false negative score for each image""",
)

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

    eval_key = brc.eval_key
    if f"{eval_key}_tp" in field_names:
        prompt += mistakenness_eval_prompt.format(eval_tp_field=f"{eval_key}_tp", eval_fp_field=f"{eval_key}_fp", eval_fn_field=f"{eval_key}_fn")
    
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
    input_variables=["text_similarity_key"],
    template=TEXT_SIMILARITY_PROMPT_TEMPLATE,
)

def generate_brain_runs_prompt(dataset, brain_runs):
    header = "Here is the relevant information about the brain runs that were run on this dataset:\n"
    prompt = header

    if "uniqueness" in brain_runs:
        uniqueness_field = brain_runs["uniqueness"]
        uniqueness_prompt = UNIQUENESS_PROMPT.format(uniqueness_field=uniqueness_field)
        prompt += uniqueness_prompt

    if "hardness" in brain_runs:
        hardness_field = brain_runs["hardness"]["hardness_field"]
        label_field = brain_runs["hardness"]["label_field"]
        hardness_prompt = HARDNESS_PROMPT.format(hardness_field=hardness_field, label_field=label_field)
        prompt += hardness_prompt

    if "image_similarity" in brain_runs:
        image_similarity_key = brain_runs["image_similarity"]
        image_similarity_prompt = IMAGE_SIMILARITY_PROMPT.format(image_similarity_key=image_similarity_key)
        prompt += image_similarity_prompt

    if "text_similarity" in brain_runs:
        text_similarity_key = brain_runs["text_similarity"]
        text_similarity_prompt = TEXT_SIMILARITY_PROMPT.format(text_similarity_key=text_similarity_key)
        prompt += text_similarity_prompt

    if "mistakenness" in brain_runs:
        mistakenness_prompt = generate_mistakenness_prompt(dataset, brain_runs["mistakenness"])
        prompt += mistakenness_prompt

    return prompt

def load_dataset_view_prompt_prefix_template():
    with open("prompts/dataset_view_generator_prefix.txt", "r") as f:
        prefix = f.read()
    return prefix

def generate_dataset_view_prompt_prefix(available_fields):
    template = load_dataset_view_prompt_prefix_template()
    prompt = PromptTemplate(
        input_variables=["available_fields"],
        template=template
    )

    return prompt.format(available_fields=available_fields)

def generate_dataset_view_prompt(
        dataset,
        required_brain_runs,
        available_fields,
        view_stage_descriptions,
        examples_prompt
    ):

    prompt = generate_dataset_view_prompt_prefix(available_fields)
    prompt += generate_brain_runs_prompt(dataset, required_brain_runs)
    prompt += view_stage_descriptions
    prompt += examples_prompt
    return prompt

def generate_dataset_view_text(
        dataset,
        required_brain_runs,
        available_fields,
        view_stage_descriptions,
        examples_prompt
    ):
    prompt = generate_dataset_view_prompt(
        dataset,
        required_brain_runs,
        available_fields,
        view_stage_descriptions,
        examples_prompt
    )

    response = llm(prompt)
    return response.strip()