"""
Label class selector.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

LABEL_CLASS_SELECTOR_PREFIX_PATH = os.path.join(
    PROMPTS_DIR, "label_class_selector_prefix.txt"
)
SEMANTIC_CLASS_SELECTOR_PREFIX_PATH = os.path.join(
    PROMPTS_DIR, "semantic_class_selector_prefix.txt"
)
LABEL_CLASS_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "fiftyone_label_class_examples.csv"
)
SEMANTIC_CLASS_SELECTOR_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "fiftyone_semantic_class_selector_examples.csv"
)

SEMANTIC_MATCH_THRESHOLD = 1000
LABELS_WITH_CLASSES = (
    "Classification",
    "Detection",
    "Polyline",
    "Classifications",
    "Detections",
    "Polylines",
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


def get_label_class_selection_examples():
    df = pd.read_csv(LABEL_CLASS_EXAMPLES_PATH)
    examples = []

    for _, row in df.iterrows():
        example = {
            "query": row.query,
            "field": row.field,
            "label_classes": row.label_classes,
        }
        examples.append(example)
    return examples


def get_semantic_class_selection_examples():
    df = pd.read_csv(SEMANTIC_CLASS_SELECTOR_EXAMPLES_PATH)
    examples = []

    for _, row in df.iterrows():
        example = {
            "class_name": row.class_name,
            "available_label_classes": row.available_label_classes,
            "semantic_matches": row.semantic_matches,
        }
        examples.append(example)
    return examples


def load_class_selector_prefix():
    with open(LABEL_CLASS_SELECTOR_PREFIX_PATH, "r") as f:
        return f.read()


def load_semantic_class_selector_prefix():
    with open(SEMANTIC_CLASS_SELECTOR_PREFIX_PATH, "r") as f:
        return f.read()


def generate_class_selector_prompt(query, label_field):
    prefix = load_class_selector_prefix()
    class_selection_examples = get_label_class_selection_examples()

    class_selection_example_formatter_template = """
    Query: {query}
    Label field: {field}
    Classes: {label_classes}\n
    """

    class_labels_prompt = PromptTemplate(
        input_variables=["query", "field", "label_classes"],
        template=class_selection_example_formatter_template,
    )

    class_selector_prompt = FewShotPromptTemplate(
        examples=class_selection_examples,
        example_prompt=class_labels_prompt,
        prefix=prefix,
        suffix="Query: {query}\nLabel field: {field}\nClasses: ",
        input_variables=["query", "field"],
        example_separator="\n",
    )

    return class_selector_prompt.format(query=query, field=label_field)


def generate_semantic_class_selector_prompt(class_name, label_classes):
    prefix = load_semantic_class_selector_prefix()
    semantic_class_selection_examples = get_semantic_class_selection_examples()

    semantic_class_selection_example_formatter_template = """
    Class name: {class_name}
    Available label classes: {available_label_classes}
    Semantic matches: {semantic_matches}\n
    """

    semantic_class_labels_prompt = PromptTemplate(
        input_variables=[
            "class_name",
            "available_label_classes",
            "semantic_matches",
        ],
        template=semantic_class_selection_example_formatter_template,
    )

    semantic_class_selector_prompt = FewShotPromptTemplate(
        examples=semantic_class_selection_examples,
        example_prompt=semantic_class_labels_prompt,
        prefix=prefix,
        suffix="Class name: {class_name}\nAvailable label classes: {available_label_classes}\nSemantic matches: ",
        input_variables=["class_name", "available_label_classes"],
        example_separator="\n",
    )

    return semantic_class_selector_prompt.format(
        class_name=class_name, available_label_classes=label_classes
    )


def identify_named_classes(query, label_field):
    class_selector_prompt = generate_class_selector_prompt(query, label_field)
    res = llm.call_as_llm(class_selector_prompt).strip()
    ncs = [c.strip().replace("'", "") for c in res[1:-1].split(",")]
    ncs = [c for c in ncs if c != ""]
    return ncs


def identify_semantic_matches(class_name, label_classes):
    semantic_class_selector_prompt = generate_semantic_class_selector_prompt(
        class_name, label_classes
    )
    res = llm.call_as_llm(semantic_class_selector_prompt).strip()
    ncs = [c.strip().replace("'", "") for c in res[1:-1].split(",")]
    ncs = [c for c in ncs if c != ""]
    ncs = [c for c in ncs if c in label_classes and c != class_name]
    if len(ncs) == 1:
        return ncs[0]
    return ncs


def get_field_type(dataset, field_name):
    sample = dataset.first()
    field = sample.get_field(field_name)
    field_type = type(field).__name__
    return field_type


def get_label_field_label_string(label_field, label_type):
    if label_type in ["Detection", "Classification", "Polyline"]:
        return f"{label_field}.label"
    else:
        return f"{label_field}.{label_type.lower()}.label"


def get_dataset_label_classes(dataset, label_field):
    field_type = get_field_type(dataset, label_field)
    field = get_label_field_label_string(label_field, field_type)
    return dataset.distinct(field)


def validate_class_name(class_name, label_classes):
    if class_name in label_classes:
        return class_name
    else:
        ## try matching with case-insensitive
        for c in label_classes:
            if c.lower() == class_name.lower():
                return c

        ## try matching with prefix
        for c in label_classes:
            if c.lower().startswith(class_name.lower()):
                return c

        return None


def select_label_field_classes(dataset, query, label_field):
    class_names = identify_named_classes(query, label_field)

    if len(class_names) == 0:
        return []
    _classes = get_dataset_label_classes(dataset, label_field)
    num_classes = len(_classes)
    sm_flag = num_classes < SEMANTIC_MATCH_THRESHOLD

    label_classes = []
    for cn in class_names:
        cn_validated = validate_class_name(cn, _classes)
        if cn_validated is not None:
            label_classes.append({cn: cn_validated})
        elif sm_flag:
            sm_classes = identify_semantic_matches(cn, _classes)
            label_classes.append({cn: sm_classes})

    return label_classes


def select_label_classes(dataset, query, fields):
    dataset_field_names = dataset.first().field_names
    present_fields = [
        f for f in fields if f != "" and f in dataset_field_names
    ]
    label_classes = {}
    for field in present_fields:
        field_type = get_field_type(dataset, field)
        if field_type in LABELS_WITH_CLASSES:
            field_label_classes = select_label_field_classes(
                dataset, query, field
            )

            if field_label_classes == "_CONFUSED_":
                return "_CONFUSED_"
            else:
                label_classes[field] = field_label_classes
    return label_classes
