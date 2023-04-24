from links.view_stage_example_selector import generate_view_stage_examples_prompt
from links.view_stage_description_selector import generate_view_stage_descriptions_prompt, get_most_relevant_view_stages
from links.algorithm_selector import select_algorithms
from links.run_selector import select_runs
from links.field_selector import select_fields
from links.label_class_selector import select_label_classes
from links.dataset_view_generator import get_gpt_view_stage_strings

def get_gpt_view_text(dataset, query):
    #### Validate media type
    if dataset.media_type not in ["image", "video"]:
        print(f"At present, the FiftyOne GPT integration only supports image and video datasets. The dataset {dataset.name} has media type {dataset.media_type}. If you would like to use this feature, please try a different dataset.")
        return
    print(f"Finding similar examples for query: {query}")
    examples = generate_view_stage_examples_prompt(
        dataset, query
        )
    view_stages = get_most_relevant_view_stages(examples)
    print(f"Identified likely view stages: {view_stages}")
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)
    algorithms = select_algorithms(query)
    if len(algorithms) > 0:
        print(f"Identified algorithms: {algorithms}")
    runs = select_runs(dataset, query, algorithms)
    run_keys = {k: v["key"] for k, v in runs.items()}
    if len(runs) > 0:
        print(f"Identified runs: {run_keys}")
    fields = select_fields(dataset, query)
    print(f"Identified potentially relevant fields: {fields}")
    label_classes = select_label_classes(dataset, query, fields)
    if len(label_classes) > 0:
        print(f"Identified label classes: {label_classes}")

    response = get_gpt_view_stage_strings(
        dataset,
        runs,
        fields,
        label_classes,
        view_stage_descriptions,
        examples
    )
    return response