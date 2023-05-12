from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

def load_effective_prompt_prefix_template():
    with open("prompts/effective_prompt_generator_prefix.txt", "r") as f:
        prefix = f.read()
    return prefix

def format_chat_history(chat_history):
    return '\n'.join(chat_history) + '\n'

def generate_dataset_view_prompt(chat_history):
    prompt = load_effective_prompt_prefix_template()
    prompt += format_chat_history(chat_history)
    prompt += "Effective prompt: "
    return prompt

def generate_effective_query(chat_history):
    prompt = generate_dataset_view_prompt(chat_history)
    response = llm.call_as_llm(prompt)
    return response.strip()
