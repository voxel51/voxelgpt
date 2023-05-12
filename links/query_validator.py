import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


def load_query_validator_prefix():
    with open("prompts/confused_task_rules.txt", "r") as f:
        prefix = f.read()
    return prefix


def get_query_validation_examples():
    df = pd.read_csv("examples/fiftyone_query_validation_examples.csv")
    examples = []

    for _, row in df.iterrows():
        example = {"input": row.input, "is_valid": row.is_valid}
        examples.append(example)
    return examples


def generate_query_validator_prompt(query):
    prefix = load_query_validator_prefix()
    validation_examples = get_query_validation_examples()

    validation_example_formatter_template = """
    Input: {input}
    Is valid: {is_valid}\n
    """

    validation_prompt = PromptTemplate(
        input_variables=["input", "is_valid"],
        template=validation_example_formatter_template,
    )

    query_validator_prompt = FewShotPromptTemplate(
        examples=validation_examples,
        example_prompt=validation_prompt,
        prefix=prefix,
        suffix="Input: {input}\nIs valid:",
        input_variables=["input"],
        example_separator="\n",
    )

    return prefix + query_validator_prompt.format(input=query)


def validate_query(query):
    prompt = generate_query_validator_prompt(query)
    res = llm.call_as_llm(prompt).strip()
    if res == "N":
        return False
    else:
        return True
