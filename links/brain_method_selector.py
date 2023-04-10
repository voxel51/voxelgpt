import pandas as pd

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
llm = OpenAI(temperature=0.0)

BRAIN_METHODS = (
        'uniqueness', 
        'image_similarity', 
        'text_similarity',
        'mistakenness',
        'hardness'
)

def get_brain_method_examples():
    df = pd.read_csv("examples/fiftyone_brain_method_examples.csv")
    examples = []

    for _, row in df.iterrows():
        methods_used = [method for method in BRAIN_METHODS if row[method] == "Y"]
        example = {
            "query": row.prompt,
            "methods": methods_used
        }
        examples.append(example)
    return examples

def load_brain_method_selector_prefix():
    with open("prompts/brain_method_selector_prefix.txt", "r") as f:
        prefix = f.read()
    return prefix

def generate_brain_method_selector_prompt(query):
    prefix = load_brain_method_selector_prefix()
    brain_method_examples = get_brain_method_examples()

    brain_methods_example_formatter_template = """
    Query: {query}
    Brain methods used: {methods}\n
    """

    brain_methods_prompt = PromptTemplate(
        input_variables=["query", "methods"],
        template=brain_methods_example_formatter_template,
    )

    brain_method_selector_prompt = FewShotPromptTemplate(
        examples=brain_method_examples,
        example_prompt=brain_methods_prompt,
        prefix=prefix,
        suffix="Query: {query}\nBrain methods used:",
        input_variables=["query"],
        example_separator="\n",
    )

    return brain_method_selector_prompt.format(query = query)

def select_brain_methods(query):
    brain_method_selector_prompt = generate_brain_method_selector_prompt(query)
    res = llm(brain_method_selector_prompt)
    return [method for method in BRAIN_METHODS if method in res]