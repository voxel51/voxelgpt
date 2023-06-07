# VoxelGPT

Wish you could search your images or videos without writing a line of code? Now
you can! 🎉

VoxelGPT is a [FiftyOne Plugin](https://docs.voxel51.com/plugins/index.html) that combines the
power of [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) with
[FiftyOne](https://github.com/voxel51/fiftyone)'s computer vision query
language, enabling you to filter, sort, and semantically slice your data with
natural language.

https://github.com/voxel51/voxelgpt/assets/12500356/3c2659a4-833d-4634-8d26-9792ab3453cd

## Live demo

🚀🚀🚀 You can try VoxelGPT live at [gpt.fiftyone.ai](https://gpt.fiftyone.ai)!

## Capabilities

VoxelGPT is capable of handling any of the following types of queries:

-   [Dataset queries](#querying-your-dataset)
-   [FiftyOne docs queries](#querying-the-fiftyone-docs)
-   [General computer vision queries](#general-computer-vision-queries)

When you ask VoxelGPT a question, it will interpret your intent and determine
which type of query you are asking. If VoxelGPT is unsure, it will ask you to
clarify.

### Querying your dataset

https://github.com/voxel51/voxelgpt/assets/12500356/5728b067-defc-4db3-8cda-ad8da3523cf4

You can ask VoxelGPT to search your datasets for you. Here's some examples of
things you can ask:

-   Retrieve me 10 random samples
-   Display the most unique images with a false positive prediction
-   Just the images with at least 2 people
-   Show me the 25 images that are most similar to the first image with a cat

Under the hood, VoxelGPT interprets your query and translates it into the
corresponding
[dataset view](https://docs.voxel51.com/user_guide/using_views.html). VoxelGPT
understands the schema of your dataset, as well as things like
[evaluation runs](https://docs.voxel51.com/user_guide/evaluation.html) and
[similarity indexes](https://docs.voxel51.com/user_guide/brain.html#similarity).
It can also automatically inspect the contents of your dataset in order to
retrieve specific entities.

### Querying the FiftyOne docs

https://github.com/voxel51/voxelgpt/assets/12500356/a32d4856-612c-4407-8dde-6bad5efb0357

VoxelGPT is not only a pair programmer; it is also an educational tool.
VoxelGPT has access to the entire [FiftyOne docs](https://docs.voxel51.com),
which it can use to answer FiftyOne-related questions.

Here's some examples of documentation queries you can ask VoxelGPT:

-   How do I load a dataset from the FiftyOne Zoo?
-   What does the match() stage do?
-   Can I export my dataset in COCO format?

### General computer vision queries

https://github.com/voxel51/voxelgpt/assets/12500356/294b53f8-9398-4e6a-b923-56c7a9684f1d

Finally, VoxelGPT can answer general questions about computer vision, machine
learning, and data science. It can help you to understand basic concepts and
learn how to overcome data quality issues.

Here's some examples of computer vision queries you can ask VoxelGPT:

-   What is the difference between precision and recall?
-   How can I detect faces in my images?
-   What are some ways I can reduce redundancy in my dataset?

## Installation

If you haven't already, install
[FiftyOne](https://github.com/voxel51/fiftyone):

```shell
pip install fiftyone
```

You'll also need to provide an OpenAI API key
([create one](https://platform.openai.com/account/api-keys)):

```shell
export OPENAI_API_KEY=XXXXXXXX
```

### App-only use

If you only want to use VoxelGPT in the
[FiftyOne App](https://docs.voxel51.com/user_guide/app.html), then you can
simply [install it as a plugin](https://docs.voxel51.com/plugins/index.html):

```shell
fiftyone plugins download https://github.com/voxel51/voxelgpt
fiftyone plugins requirements @voxel51/voxelgpt --install
```

### Local use/development

If you want to directly use the `voxelgpt` module or develop the project
locally, then you'll want to clone repository:

```shell
git clone https://github.com/voxel51/voxelgpt
cd voxelgpt
```

install the requirements:

```shell
pip install -r requirements.txt
```

and make the plugin available for use in the FiftyOne App by symlinking it into
your plugins directory:

```shell
# Symlinks your clone of voxelgpt into your FiftyOne plugins directory
ln -s "$(pwd)" "$(fiftyone config plugins_dir)/voxelgpt"
```

### FiftyOne Teams

Want to add VoxelGPT to your
[FiftyOne Teams](https://voxel51.com/fiftyone-teams) deployment? You can!
[Instructions here](FIFTYONE_TEAMS.md).

## Using VoxelGPT in the App

You can use VoxelGPT in the FiftyOne App by loading any dataset:

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

and then either:

-   Clicking on the OpenAI icon above the grid
-   Pressing the `+` icon next to the Samples tab and choosing VoxelGPT

https://github.com/voxel51/voxelgpt/assets/12500356/fbaccb6f-dc7f-43d7-9da3-adb4062c398b

For example, try asking the following questions:

-   What are some popular model architectures for computer vision?
-   How do I search for specific objects in my dataset?
-   Show me predicted airplanes

**Pro tip:** use the [`now` keyword](#keywords) to incorporate your previous
prompts as context for your next query!

You can also run VoxelGPT as an
[operator](https://docs.voxel51.com/plugins/index.html#fiftyone-operators) by
pressing the `~` key on your keyboard and selecting `Ask VoxelGPT` from the
list. This will open up a small modal where you can type in your query.

## Using VoxelGPT in Python

If you've [installed locally](local-use/development), you can also directly
interact with VoxelGPT via Python.

### Interactive sessions

You can use `ask_voxelgpt_interactive()` to launch an interactive session where
you can converse with VoxelGPT via `input()` prompts:

```py
import fiftyone as fo
import fiftyone.zoo as foz

from voxelgpt import ask_voxelgpt_interactive

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)

ask_voxelgpt_interactive(dataset, session=session)
```

**Pro tip:** use the [`now` keyword](#keywords) to incorporate your previous
prompts as context for your next query.

As usual, you can prompt VoxelGPT with any combination of dataset,
documentation, and general computer vision queries. For example, a conversation
might look like:

```
You:
    what is the difference between precision and recall?

VoxelGPT:
    Precision and recall are two important metrics used to evaluate the
    performance of a machine learning model in classification tasks. Precision
    measures the proportion of true positive predictions among all positive
    predictions made by the model. Recall, on the other hand, measures the
    proportion of true positive predictions among all actual positive instances
    in the dataset.

    In other words, precision is the ability of the model to correctly identify
    positive instances, while recall is the ability of the model to identify
    all positive instances. A high precision means that the model is making
    very few false positive predictions, while a high recall means that the
    model is able to identify most of the positive instances in the dataset.

    It is important to balance precision and recall when evaluating a model's
    performance. A model with high precision but low recall may be missing
    important positive instances, while a model with high recall but low
    precision may be making too many false positive predictions.

    In the context of the FiftyOne library, precision and recall can be
    computed using the `fiftyone.utils.metrics` module, which provides
    functions for computing various classification metrics. For example, the
    `fiftyone.utils.metrics.binary_classification` function can be used to
    compute precision and recall for binary classification tasks.

You:
    does FiftyOne integrate with Label Studio?

VoxelGPT:
    Yes, FiftyOne integrates with Label Studio. The integration allows you to
    easily upload your data directly from FiftyOne to Label Studio for labeling.
    You can get started with Label Studio through a simple pip install to get a
    local server up and running. FiftyOne provides simple setup instructions
    that you can use to specify the necessary account credentials and server
    endpoint to use. FiftyOne provides an API to create projects, upload data,
    define label schemas, and download annotations using Label Studio, all
    programmatically in Python. All of the following label types are supported
    for image datasets: Classification, Detections, Instance segmentations,
    Polygons and polylines, Keypoints, Scalar fields, and Semantic segmentation.

    Sources:
    - https://docs.voxel51.com/integrations/index.html
    - https://docs.voxel51.com/integrations/labelstudio.html

You:
    show me predicted airplanes

VoxelGPT:
    Identified potential algorithms: evaluation
    No evaluation runs found.
    Identified potential fields: predictions
    Identified potential label classes: {'predictions': ['airplane']}
    Identified potential view stages: ['match', 'filter_labels', 'match_labels', 'exclude_labels', 'filter_field']
    Okay, I'm going to load dataset.filter_labels("predictions",F("label") == "airplane")

You:
    now only show me the first 10 samples

VoxelGPT:
    Identified potential algorithms: evaluation
    No evaluation runs found.
    Identified potential fields: predictions
    Identified potential label classes: {'predictions': ['airplane']}
    Identified potential view stages: ['match', 'limit', 'limit_labels', 'skip', 'sort_by']
    Okay, I'm going to load dataset.match(F("predictions.detections.label").contains("airplane")).limit(10)

You: exit
```

In interactive mode, VoxelGPT automatically loads any views it creates in the
App, and you can access them via your
[session](https://docs.voxel51.com/user_guide/app.html#sessions) object:

```py
print(session.view.count("predictions.detections"))
```

### Single queries

You can also use `ask_voxelgpt()` to prompt VoxelGPT with individual queries:

```py
from voxelgpt import ask_voxelgpt

ask_voxelgpt("Does FiftyOne integrate with CVAT?")
```

```

Yes, FiftyOne integrates with CVAT, which is an open-source image and video
annotation tool. You can upload your data directly from FiftyOne to CVAT to add or
edit labels. You can use CVAT either through the hosted server at app.cvat.ai or
through a self-hosted server. In either case, FiftyOne provides simple setup
instructions that you can use to specify the necessary account credentials and
server endpoint to use. The tight integration between FiftyOne and CVAT allows
you to curate and explore datasets in FiftyOne and then send off samples or
existing labels for annotation in CVAT with just one line of code. To use CVAT,
you must create an account on a CVAT server. By default, FiftyOne uses app.cvat.ai.
If you haven’t already, go to app.cvat.ai and create an account now. Another option
is to set up CVAT locally and then configure FiftyOne to use your self-hosted server.
A primary benefit of setting up CVAT locally is that you are limited to 10 tasks and
500MB of data with app.cvat.ai.

Sources:
- https://docs.voxel51.com/integrations/cvat.html#examples
- https://docs.voxel51.com/tutorials/cvat_annotation.html#Annotating-Datasets-with-CVAT
- https://docs.voxel51.com/tutorials/cvat_annotation.html#Setup
- https://docs.voxel51.com/integrations/index.html#fiftyone-integrations
```

When VoxelGPT creates a view in response to your query, it is returned:

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")

view = ask_voxelgpt("show me 10 random samples", dataset)
```

```
Identified potential view stages: ['match', 'limit', 'skip', 'take', 'sort_by']
Okay, I'm going to load dataset.take(10)
```

## Keywords

VoxelGPT is trained to recognize certain keywords that help it understand your
intent:

| Keyword                 | Meaning                                                                                                                                                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `show`/`display`        | Tells VoxelGPT that you want it to query your dataset and display the results                                                                                                                                                                                |
| `docs`/`how`/`FiftyOne` | Tells VoxelGPT that you want it to query the FiftyOne docs.                                                                                                                                                                                                  |
| `now`                   | Use your chat history as context to interpret your next query. For example, if you ask "show me images with people" and then ask "now show me the 10 most unique ones", VoxelGPT will understand that you want to show the 10 most unique images with people |
| `help`                  | Prints a help message with usage instructions                                                                                                                                                                                                                |
| `reset`                 | Resets the conversation history                                                                                                                                                                                                                              |
| `exit`                  | Exits interactive Python sessions                                                                                                                                                                                                                            |

## Contributing

Contributions are welcome! Check out the [contributions guide](CONTRIBUTING.md)
for instructions.

## How does it work?

VoxelGPT uses:

-   OpenAI's [GPT-3.5 model](https://platform.openai.com/docs/models/gpt-3-5)
    to generate answers, including Python code
-   OpenAI's
    [text-embedding-ada-002 model](https://platform.openai.com/docs/guides/embeddings/embedding-models)
    to embed input text prompts
-   [LangChain](https://github.com/hwchase17/langchain) provides the connective
    tissue for the application
-   FiftyOne's [plugin framework](https://docs.voxel51.com/plugins/index.html)
    to provide the interactive panel in the
    [FiftyOne App](https://docs.voxel51.com/user_guide/app.html)

## Limitations

### Media types

VoxelGPT currently only supports image datasets. We're working on adding
support for videos and other media types.

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

If you've made it this far, we'd greatly appreciate if you'd take a moment to
check out [FiftyOne](https://github.com/voxel51/fiftyone) and give us a star!

FiftyOne is an open source library for building high-quality datasets and
computer vision models. It's the engine that powers this project.

Thanks for visiting! 😊

## Join the Community

If you want join a fast-growing community of engineers, researchers, and
practitioners who love computer vision, join the
[FiftyOne Slack community](https://slack.voxel51.com/)! 🚀🚀🚀
