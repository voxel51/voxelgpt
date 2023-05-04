import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

def get_label_class_selection_examples():
    df = pd.read_csv("examples/fiftyone_label_class_examples.csv")
    examples = []

    for _, row in df.iterrows():
        example = {
            "query": row.query,
            "field": row.field,
            "label_classes": row.label_classes
        }
        examples.append(example)
    return examples

LABELS_WITH_CLASSES = (
    "Classification",
    "Detection",
    "Polyline",
    "Classifications",
    "Detections",
    "Polylines",
)

def load_class_selector_prefix():
    with open("prompts/label_class_selector_prefix.txt", "r") as f:
        prefix = f.read()
    return prefix


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

    return class_selector_prompt.format(
        query = query, 
        field = label_field
        )

def identify_named_classes(query, label_field):
    class_selector_prompt = generate_class_selector_prompt(query, label_field)
    res = llm.call_as_llm(class_selector_prompt).strip()
    ncs = [c.strip().replace('\'', '') for c in res[1:-1].split(",")]
    ncs = [c for c in ncs if c != ""]
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

def validate_class_name(class_name, label_classes, label_field):
    if class_name in label_classes:
        return class_name
    else:
        ## try matching with case-insensitive
        for c in label_classes:
            if c.lower() == class_name.lower():
                print(f"Matching {class_name} with {c}")
                return c
        
        ## try matching with prefix
        for c in label_classes:
            if c.lower().startswith(class_name.lower()):
                print(f"Matching {class_name} with {c}")
                return c
        print(f"Class name {class_name} not found for label {label_field}")
        return class_name

def select_label_field_classes(dataset, query, label_field):
    class_names = identify_named_classes(query, label_field)
    if len(class_names) == 0:
        return []
    _classes = get_dataset_label_classes(dataset, label_field)

    ### Need better way to match class names
    # labels_classes = []
    # for cn in class_names:
        ## if fewer than 1k classes, try semantic matching --> need new prompt
        ## if more than 1k classes, don't try semantic matching

        # if nothing found, and no text similarity run, then ask for more info

    label_classes = [
        validate_class_name(c, _classes, label_field) 
        for c in class_names
        ]
    
    return label_classes

def select_label_classes(dataset, query, required_fields):
    field_names = dataset.first().field_names
    fields = [f.strip() for f in required_fields[1:-1].split(",")]
    fields = [f for f in fields if f != "" and f in field_names]
    label_classes = {}
    for field in fields:
        field_type = get_field_type(dataset, field)
        if field_type in LABELS_WITH_CLASSES:
            label_classes[field] = select_label_field_classes(
                dataset, 
                query, 
                field
                )
    return label_classes