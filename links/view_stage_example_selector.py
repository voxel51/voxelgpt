import pandas as pd
import re

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

EMBEDDINGS = OpenAIEmbeddings()

VIEW_STAGE_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

def format_example_text(text):
    text = text.replace("\"\"", "'").replace("\"\'", "'")
    text = text.replace("\'\"", "'").replace("\'\'", "'").replace("\"", "'")
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    text = re.sub('\s{2,}', ' ', text).replace("\"", "'")
    return text

def get_view_stage_examples(dataset):
    media_type = dataset.media_type
    examples = pd.read_csv("examples/fiftyone_viewstage_examples.csv")
    relevant_examples = examples[examples["media_type"].isin([media_type, "all"])][["query", "stages"]]
    queries = relevant_examples["query"].tolist()
    stages_lists = relevant_examples["stages"].tolist()
    examples_dict = [
        {"input": format_example_text(query), "output": format_example_text(sl)} 
        for query, sl in zip(queries, stages_lists)]
    return examples_dict

def generate_view_stage_example_selector(dataset):
    examples = get_view_stage_examples(dataset)
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples, 
        EMBEDDINGS, 
        Chroma, 
        k=10,
    )
    return example_selector

def generate_view_stage_examples_prompt_template(dataset):
    example_selector = generate_view_stage_example_selector(dataset)
    example_prompt = VIEW_STAGE_EXAMPLE_PROMPT
    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Generate code to produce the FiftyOne view stages for the following prompts:\n",
        suffix="Input: {text}\nOutput:", 
        input_variables=["text"],
    )

def generate_view_stage_examples_prompt(dataset, query):
    similar_examples_prompt_template = generate_view_stage_examples_prompt_template(dataset)
    return similar_examples_prompt_template.format(text = query)
