from links.view_stage_example_selector import generate_view_stage_examples_prompt
from links.view_stage_description_selector import generate_view_stage_descriptions_prompt, get_most_relevant_view_stages
from links.brain_method_selector import select_brain_methods
from links.brain_run_selector import select_brain_runs
from links.field_selector import select_fields
from links.dataset_view_generator import generate_dataset_view_text

def get_gpt_view_text(dataset, query):
    print(f"Finding similar examples for query: {query}")
    examples = generate_view_stage_examples_prompt(
        dataset, query
        )
    view_stages = get_most_relevant_view_stages(examples)
    print(f"Identified likely view stages: {view_stages}")
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)
    brain_methods = select_brain_methods(query)
    print(f"Identified brain methods: {brain_methods}")
    brain_runs = select_brain_runs(dataset, query, brain_methods)
    print(f"Identified brain runs: {brain_runs}")
    fields = select_fields(dataset, query)
    print(f"Identified fields: {fields}")

    response = generate_dataset_view_text(
        dataset,
        brain_runs,
        fields,
        view_stage_descriptions,
        examples
    )
    return response