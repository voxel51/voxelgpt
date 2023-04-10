# GPT View Stages
Goal: use GPT model to interpret user queries and generate valid view stages!

## Relevant Links
- Examples: https://docs.google.com/spreadsheets/d/14pdiODt9PmsD2F___Lpv2rO-RByTtC8PYcEfxL8iIuY/edit#gid=186467173
- LangChain: https://github.com/hwchase17/langchain
- Chroma: no docker or API_KEY - https://www.trychroma.com/
- OpenAI account usage: https://platform.openai.com/account/usage

## Tasks

- **Eric & Jacob**: Validation (`validate_stages.py` is not yet working, and has not been incorporated)
- **Jacob**: Label class recognition
    - determine what classes the user is referring to for each label field
- **Allen**: Evaluation Identification Stage:
    - `links/evaluation_run_selector.py` 
    - add stage to `dataset_view_generator.py`
    - add `prompts/evaluation_task_rules.txt` prompt
    - add examples to Examples spreadsheet in new tab, and then put in `examples` folder
- **Leila**: Add support for `hardness` brain runs
    - fill out the template in `links/brain_run_selector.py` 
    - add `prompts/hardness_task_rules.txt` prompt modeled after uniqueness and mistakenness.
    - new examples tab `hardness` modeled after `uniqueness` and `mistakenness` tabs and then put in `examples` folder
- **Vini**: More examples
    - scrape examples from community Slack covering as wide a range of scenarios as possible
    - complex filters and view expressions!
    - videos
    - every type of label
    - varied naming conventions

## Additional notes

- Want to validate embedded fields
- At present, there is no support for multi-line Python code to generate views. It all needs to be done inline
- No support for groups, 3D/point clouds, `concat()`, or `mongo()`.

## Getting started

`pip install openai langchain chroma`

Then create an OpenAI account and generate an API key

`export OPENAI_API_KEY=...`