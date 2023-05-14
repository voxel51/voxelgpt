# ChatGPT <> FiftyOne Integration

Wish you could search your images or videos without writing a line of code? Now
you can!

This integration combines the power of
[GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) with
[FiftyOne](https://github.com/voxel51/fiftyone)'s computer vision query
language, enabling you to filter, sort, and semantically slice your data with
natural language.

## Try it live

You can test drive this integration live at
[gpt.fiftyone.ai](https://gpt.fiftyone.ai).

Here's some examples of things you can ask ChatGPT to do:

- Show me the most unique images with a false positive prediction
- Retrieve the first 10 images with 3 dogs and 1 cat
- Just display objects with small bounding boxes
- Random sampling of images where AlexNet and Inception v3 disagree

## Installation

If you want to run this locally, you'll need to:

1. Clone the repository:

```shell
git clone https://github.com/voxel51/fiftyone-gpt
cd fiftyone-gpt
```

2. Install some packages:

```shell
pip install openai langchain chromadb pandas
```

3. Provide your OpenAI API key
   ([create one](https://platform.openai.com/account/api-keys)):

```shell
export OPENAI_API_KEY=XXXXXXXX
```

## Try it locally

```py
import fiftyone as fo
import fiftyone.zoo as foz

from gpt_view_generator import ask_gpt_interactive

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

```py
ask_gpt_interactive(dataset, session=session)
# How can I help you? show me 10 random samples
```

```
Getting or creating embeddings for queries...
Loading embeddings from file...
Saving embeddings to file...
Identified likely view stages: ['limit', 'skip', 'take', 'match', 'sort_by_similarity']
Identified potentially relevant fields: ['']
Did not identify any relevant label classes
Stage 1: take(10)
```

## Contributing

If you plan to contribute a PR, please install the pre-commit hooks before
commiting:

```shell
pre-commit install
```

You can manually lint a file if necessary like so:

```shell
# Manually run linting configured in the pre-commit hook
pre-commit run --files <file>
```

## How does it work?

- OpenAI's
  [text-embedding-ada-002 model](https://platform.openai.com/docs/guides/embeddings/embedding-models)
  is used to embed the input text prompts
- [Chroma](https://www.trychroma.com) is used to perform in-memory vector
  searches for the most similar examples to input text prompts
- LangChain](https://github.com/hwchase17/langchain) provides the connective
  tissue for the application
- OpenAI's [GPT-3.5 model](https://platform.openai.com/docs/models/gpt-3-5)
  (the model underpinning ChatGPT) is used to generate the Python code that
  is executed by [FiftyOne](https://github.com/voxel51/fiftyone)
- The [FiftyOne App](https://docs.voxel51.com/user_guide/app.html) is used to
  display the results of the generated Python code

## Limitations

### Media types

This MVP implementation only supports images. We're working on adding support
for videos and other media types.

### Examples

This MVP implementation is based on a limited set of examples, so it may not
generalize well to your data. The more specific your query, the better the
results will be. If you find that the results are not what you expect, please
let us know!

### Interactivity

The current implementation is not interactive. You can't ask follow-up
questions or refine your query. We're working on it!

### View stages

While the current implementation supports most of FiftyOne's `ViewStage`
methods, it does not support all of them. In particular, it does not support
`concat()`, `mongo()`, or `geo_within()`. Again, we're working on it!

## About FiftyOne

If you've made it this far, we'd greatly appreciate if you'd take a moment to
check out [FiftyOne](https://github.com/voxel51/fiftyone) and give us a star!

FiftyOne is an open source library for building high-quality datasets and
computer vision models. It's the engine that powers this project.

Thanks for using :)
