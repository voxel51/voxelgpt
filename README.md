# VoxelGPT

Wish you could search your images or videos without writing a line of code? Now you can! 🎉

VoxelGPT combines the power of
[GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) with
[FiftyOne](https://github.com/voxel51/fiftyone)'s computer vision query
language, enabling you to filter, sort, and semantically slice your data with
natural language.

## Capabilities

VoxelGPT is capable of handling any of the following types of queries:

- [Dataset queries](#query-your-dataset)
- [FiftyOne docs queries](#query-the-fiftyone-docs)
- [Computer vision queries](#computer-vision-queries)

When you ask VoxelGPT a question, it will interpret your intent, and determine which type of query you are asking. If VoxelGPT is unsure, it will ask you to clarify.

### Query your dataset

VoxelGPT can interpret your query, translate it into FiftyOne query language Python code, and display the resulting view. VoxelGPT understands the schema of your dataset, so it can help you write queries that are valid for your dataset.

Here's some examples of dataset queries you can ask VoxelGPT - try them out live at [try.fiftyone.ai](https://try.fiftyone.ai):

-  Retrieve me 10 random samples
-  Display the most unique images with a false positive prediction
-  Just the images with at least 2 people
-  Show me the 25 images that are most similar to the first image with a cat

### Query the FiftyOne docs

VoxelGPT is not just a pair programmer; it is also an educational tool. The model has access to the entire FiftyOne documentation, and can use this to answer questions.

Here's some examples of documentation queries you can ask VoxelGPT

- How do I load a dataset from the FiftyOne Zoo?
- What does the match() stage do?
- Can I export my dataset in COCO format?

### Computer vision queries

Finally, VoxelGPT can answer general questions in computer vision, machine learning, and data science. It can help you to understand basic concepts and overcome data quality issues.

Here's some examples of computer vision queries you can ask VoxelGPT:

- What is the difference between precision and recall?
- How can I detect faces in my images?
- What are some ways I can reduce redundancy in my dataset?

## How to use

### VoxelGPT Python API

After installing (see below), you can use VoxelGPT in Python.

There are two modes of operation: single-query mode, and interactive mode.

#### Single-query mode

In single-query mode, you provide a query and VoxelGPT will return the results. You can use VoxelGPT in single-query mode via the `ask_voxelgpt` function:

```py
import fiftyone as fo
import fiftyone.zoo as foz

from voxelgpt import ask_voxelgpt

dataset = foz.load_zoo_dataset("quickstart")
query = "show me 10 random samples"
view = ask_voxelgpt_interactive(query, dataset)
```

This will return a view that contains the results of your query. For computer vision queries or FiftyOne docs queries, which do not require a dataset, you can pass in `None` for the second argument.

```py
query = "Does FiftyOne integrate with Label Studio?"
ask_voxelgpt_interactive(query, None)
```

```
FILL IN
```

#### Interactive mode

In interactive mode, you can have a conversation with VoxelGPT. You can pass VoxelGPT any combination of dataset, documentation, and general computer vision queries. When your query includes the `now` keyword, VoxelGPT will use your chat history to generate an effective query, encapsulating what it thinks you are asking for.

You can use VoxelGPT in interactive mode via the `ask_voxelgpt_interactive` function. Because this is


```py
import fiftyone as fo
import fiftyone.zoo as foz

from voxelgpt import ask_voxelgpt_interactive

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

```py
ask_voxelgpt_interactive(dataset, session=session)
```

```
>> How can I help you? show me 10 random samples

# Identified potential view stages: ['sort_by', 'limit', 'skip', 'take', 'exclude']
# Okay, I'm going to load dataset.take(10)

>> Now sort these by uniqueness

# FILL IN
```

### VoxelGPT Plugin

Optionally, you can install VoxelGPT as a plugin for FiftyOne, which will allow you to use VoxelGPT natively within the FiftyOne App!

If you have the plugin installed, there are three ways to access VoxelGPT.

Launch a session of the FiftyOne App: 

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Then you can use VoxelGPT in the following ways:

1. Click on the OpenAI icon in the menu bar to open up a VoxelGPT panel. If you want to query your dataset, click on either the vertical split or horizontal split icons to split the canvas of the FiftyOne App into two panels. The `Samples` panel is attached to the `VoxelGPT` panel, so it will automatically update as you query your dataset.
2. You can also open the VoxelGPT panel by clicking on the `+` icon to the right of the `Samples` tab and selecting `VoxelGPT` from the dropdown menu.
3. If you want to run a standalone query, and don't want to open up a new panel, you can use VoxelGPT as an "operator" by pressing the `~` key on your keyboard and selecting `Ask VoxelGPT` from the list. This will open up a small modal where you can type in your query. When you press `Enter`, the results of your query will be displayed in the `Samples` panel.


### Keywords

VoxelGPT is trained to recognize certain keywords that help it to understand your intent. These keywords are:

- `show`/`display`: These keywords tell VoxelGPT that you want it to query your dataset and display the results.
- `now`: VoxelGPT keeps track (locally) of the conversation between the user and the model. This keyword tells VoxelGPT to use the chat history along with your most recent input to generate an effective query. For example, if you ask VoxelGPT "show me images with people", and then ask "now show me the 10 most unique ones", VoxelGPT will understand that you want to show the 10 most unique images with people.
- `reset`: This keyword resets the conversation history.
- `exit`: This keyword exits the program.

## Live demo

You can try VoxelGPT out live at [try.fiftyone.ai](https://try.fiftyone.ai).



## Installation

If you want to run this locally, you'll need to:

1. Clone the repository:

```shell
git clone https://github.com/voxel51/voxelgpt
cd voxelgpt
```

2. Install these packages:

```shell
pip install openai langchain chromadb pandas
```

3. Provide your OpenAI API key
   ([create one](https://platform.openai.com/account/api-keys)):

```shell
export OPENAI_API_KEY=XXXXXXXX
```

### VoxelGPT Plugin

If you want to install VoxelGPT as a plugin for FiftyOne, you can do so by running:

FILL IN




## Contributing

We welcome contributions to this project! If you plan to contribute a PR, please install the pre-commit hooks before
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

-   OpenAI's
    [text-embedding-ada-002 model](https://platform.openai.com/docs/guides/embeddings/embedding-models)
    is used to embed the input text prompts
-   [Chroma](https://www.trychroma.com) is used to perform in-memory vector
    searches for the most similar examples to input text prompts
-   LangChain](https://github.com/hwchase17/langchain) provides the connective
    tissue for the application
-   OpenAI's [GPT-3.5 model](https://platform.openai.com/docs/models/gpt-3-5)
    (the model underpinning ChatGPT) is used to generate the Python code that
    is executed by [FiftyOne](https://github.com/voxel51/fiftyone)
-   The [FiftyOne App](https://docs.voxel51.com/user_guide/app.html) is used to
    display the results of the generated Python code

## Limitations

### Media types

This MVP implementation only supports images. We're working on adding support
for videos and other media types.

### Examples

This implementation is based on a limited set of examples, so it may not
generalize well to all datasets. The more specific your query, the better the
results will be. If you find that the results are not what you expect, please
let us know!

### View stages

The current implementation supports most FiftyOne
[view stages](https://docs.voxel51.com/user_guide/using_views.html), but
certain stages like `concat()`, `mongo()`, and `geo_within()` are not yet
supported. We're working on it!

## About FiftyOne

If you've made it this far, we'd greatly appreciate if you'd take a moment to check out [FiftyOne](https://github.com/voxel51/fiftyone) and give us a star!

FiftyOne is an open source library for building high-quality datasets and computer vision models. It's the engine that powers this project.

Thanks for visiting! 😊

## Join the Community
If you want to be a part of our fast-growing community of engineers, researchers, and practitioners in machine learning and computer vision, join our the [FiftyOne Community Slack](https://join.slack.com/t/fiftyone-users/shared_invite/zt-s6936w7b-2R5eVPJoUw008wP7miJmPQ)!


## License
The VoxelGPT project is released under the Apache 2.0 license.