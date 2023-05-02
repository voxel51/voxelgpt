# Filter your images with ChatGPT

Have you ever wanted to dig into your images or videos without writing a line of code? Now you can!

We've combined the power of [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) with [FiftyOne](https://github.com/voxel51/fiftyone)'s computer vision query language so that you can filter, sort, and semantically slice your data with natural language. 

## Example queries

- Show me the most unique images with a false positive prediction
- Retrieve the first 10 images with 3 dogs and 1 cat
- Just display objects with small bounding boxes
- Random sampling of images where AlexNet and Inception v3 disagree

## Try it out at [gpt.fiftyone.ai](https://gpt.fiftyone.ai)!
This site is hosted by [Voxel51](https://voxel51.com/), and is free to use. No OpenAI API key is required!

## Installation

If you want to run this locally, you'll need to install the following:

1. Clone the `fiftyone-gpt` repo

```shell
git clone https://github.com/voxel51/fiftyone-gpt
```

2. Install the `fiftyone-gpt` package by `cd`ing into the repo and running:

```shell
pip install -e .
```

3. [register an API key](https://platform.openai.com/account/api-keys). Once you have your API key, set the `OPENAI_API_KEY` environment variable to it:

```shell
export OPENAI_API_KEY=<your key>
```

4. Something about plugin?

## How does it work?

- OpenAI's [text-embedding-ada-002 model](https://platform.openai.com/docs/guides/embeddings/embedding-models) is used to embed the input text prompts.
- Vector database [Chroma](https://www.trychroma.com/) is used to perform in-memory searches for the most similar examples to input text prompts.
- Open source library [LangChain](https://github.com/hwchase17/langchain) provides the connective tissue for the application. 
- OpenAI's [GPT-3.5 model](https://platform.openai.com/docs/models/gpt-3-5) (the model underpinning of ChatGPT) is used to generate the Python code that is executed by [FiftyOne](https://github.com/voxel51/fiftyone).
- The [FiftyOne App](https://docs.voxel51.com/user_guide/app.html) is used to display the results of the generated Python code.

## Limitations

### Media types

This MVP implementation only supports images. We're working on adding support for videos and other media types.

### Examples

This MVP implementation is based on a limited set of examples, so it may not generalize well to your data. The more specific your query, the better the results will be. If you find that the results are not what you expect, please let us know!

### Interactivity

The current implementation is not interactive. You can't ask follow-up questions or refine your query. We're working on it!

### View stages

While the current implementation supports most of FiftyOne's `ViewStage` methods, it does not support all of them. In particular, it does not support `concat()`, `mongo()`, or `geo_within()`. Again, we're working on it!

## Main repo

If you've made it this far, we'd greatly appreciate if you'd take a moment to check out the [FiftyOne](https://github.com/voxel51/fiftyone) repo and give that project a star. FiftyOne is an open source library for building high-quality datasets and computer vision models. It's the engine that powers this project! Thanks so much :)