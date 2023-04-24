from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')


def load_query_validator_prefix():
    with open('prompts/confused_task_rules.txt', 'r') as f:
        prefix = f.read()
    return prefix

def generate_query_validator_prompt(query):
    prefix = load_query_validator_prefix()

    prompt_template = """
    input: {query}
    output:
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template,
    ).format(query=query)

    return prefix + prompt

def validate_query(query):
    print('validating query')
    prompt = generate_query_validator_prompt(query)
    res = llm.call_as_llm(prompt).strip()
    if res == 'N':
        return False
    else:
        return True