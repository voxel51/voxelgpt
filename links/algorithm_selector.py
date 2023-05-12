import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

ALGORITHMS = (
    "uniqueness",
    "image_similarity",
    "text_similarity",
    "mistakenness",
    "hardness",
    "evaluation",
    "metadata",
)


def get_algorithm_examples():
    df = pd.read_csv("examples/fiftyone_algorithm_examples.csv")
    examples = []

    for _, row in df.iterrows():
        algorithms_used = [alg for alg in ALGORITHMS if row[alg] == "Y"]
        example = {"query": row.prompt, "algorithms": algorithms_used}
        examples.append(example)
    return examples


def load_algorithm_selector_prefix():
    with open("prompts/algorithm_selector_prefix.txt", "r") as f:
        prefix = f.read()
    return prefix


def generate_algorithm_selector_prompt(query):
    prefix = load_algorithm_selector_prefix()
    algorithm_examples = get_algorithm_examples()

    algorithm_example_formatter_template = """
    Query: {query}
    Algorithms used: {algorithms}\n
    """

    algorithms_prompt = PromptTemplate(
        input_variables=["query", "algorithms"],
        template=algorithm_example_formatter_template,
    )

    algorithm_selector_prompt = FewShotPromptTemplate(
        examples=algorithm_examples,
        example_prompt=algorithms_prompt,
        prefix=prefix,
        suffix="Query: {query}\nAlgorithms used:",
        input_variables=["query"],
        example_separator="\n",
    )

    return algorithm_selector_prompt.format(query=query)


def select_algorithms(query):
    algorithm_selector_prompt = generate_algorithm_selector_prompt(query)
    res = llm.call_as_llm(algorithm_selector_prompt)
    return [alg for alg in ALGORITHMS if alg in res]
