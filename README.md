# VoxelGPT

Wish you could search your images or videos without writing a line of code?
Want to extract insights from your data by asking in plain English? Now you
can! 🎉

VoxelGPT is a [FiftyOne Plugin](https://docs.voxel51.com/plugins/index.html)
that combines the power of large language models (LLMs) and large multimodal
models (LMMs) with [FiftyOne](https://github.com/voxel51/fiftyone)'s computer
vision query language, enabling you to filter, sort, semantically slice and ask
questions about your data using natural language. It can even perform
computations on your dataset for you — with approval, of course!

## Live demo

🚀🚀🚀 You can try VoxelGPT live at [gpt.fiftyone.ai](https://gpt.fiftyone.ai)!

## Capabilities

VoxelGPT is capable of handling any of the following types of queries:

-   [Dataset queries](#querying-your-dataset)
-   [Computation queries](#computational-queries)
-   [FiftyOne Library queries](#fiftyone-library-queries)
-   [FiftyOne Workspace queries](#querying-your-workspace)
-   [General computer vision or machine learning queries](#general-computer-vision-queries)

When you ask VoxelGPT a question, it will interpret your intent and determine
which type of query you are asking. If VoxelGPT is unsure, it will ask you to
clarify.

### Querying your dataset

https://github.com/voxel51/voxelgpt/assets/12500356/5728b067-defc-4db3-8cda-ad8da3523cf4

VoxelGPT can handle the following types of queries about your dataset:

-   Answer questions about the schema of your dataset, fields, and runs that
    have been performed
-   Create a filtered view of your data by constructing and concatenating view
    stages.
-   Set the view in the FiftyOne App
-   Perform aggregations over the entire dataset or a view into the dataset

You can ask VoxelGPT to search your datasets for you. Here's some examples of
things you can ask:

-   Show me 10 random samples
-   Show me high confidence false positive predictions
-   Do I have any images with multiple people?
-   What is the average brightness for my images that contain a cat?

Under the hood, VoxelGPT interprets your query and translates it into the
corresponding
[dataset view](https://docs.voxel51.com/user_guide/using_views.html). VoxelGPT
understands the schema of your dataset, as well as things like
[evaluation runs](https://docs.voxel51.com/user_guide/evaluation.html) and
[similarity indexes](https://docs.voxel51.com/user_guide/brain.html#similarity).

It can also automatically inspect the contents of your dataset in order to
retrieve specific entities.

#### Data Schema Queries

VoxelGPT can answer questions about the schema of your dataset, brain runs,
evaluation runs, and more. Here are some examples:

-   What fields do I have in my dataset?
-   Do I have any evaluation runs?
-   What model did I use to similarity index my dataset?

#### Object Detection Queries

If your dataset contains one or more
[`fo.Detections`](https://docs.voxel51.com/user_guide/using_datasets.html#object-detection)
field(s), VoxelGPT can filter or match based on the size (relative and
absolute) of bounding boxes, and on the number of detections.

-   Restrict the view to ground truth detections larger than half of the image
    area
-   Show me all of the predictions < $96^2$ pixels
-   What is the average number of person detections I have per image?

#### Geolocation Queries

If your dataset has a
[`GeoLocation`](https://docs.voxel51.com/user_guide/using_datasets.html#geolocation)
field, you can run geographic queries on your dataset. VoxelGPT can perform
geocoding to go from location name (or textual description) to a `(lon, lat)`
pair, or a list of `(lon, lat)` points defining a boundary region. Here are a
few examples:

-   Sort by proximity to the Statue of Liberty
-   Show me samples within 400m of Grand Central
-   Filter for images of Paris
-   How many images do I have that were taken in Hell's Kitchen?

#### Temporal Queries

If your dataset has a
[`Date` or `DateTime` field](https://docs.voxel51.com/user_guide/basics.html#fields),
VoxelGPT can perform temporal queries such as:

-   Filter for pictures taken on a Tuesday
-   How many images were added after June 01, 2023?
-   Show me samples with `event` field reading a time of day between 8pm and
    11pm

#### Aggregations

VoxelGPT has access to Aggregation stages in FiftyOne, so it can perform
aggregations like `count`, `mean`, `sum`, `std`, `min`, `max`, `values`, and
`distinct` for a field or expression over the entire dataset or a view into the
dataset. Here are some examples:

-   What is the average brightness of my images?
-   How many images do I have with a `cat` label?
-   What is the standard deviation of the `confidence` field in my predictions?

### Computational queries

VoxelGPT can perform computations on your dataset, such as:

-   brightness: assign a brightness score to each sample in the dataset, using
    FiftyOne's
    [Image Quality Issues plugin](https://github.com/jacobmarks/image-quality-issues)
-   entropy: quantify the amount of information in each sample in the dataset,
    using FiftyOne's
    [Image Quality Issues plugin](https://github.com/jacobmarks/image-quality-issues)
-   uniqueness: assign a uniqueness score to each sample in the dataset, using
    the [FiftyOne Brain](https://voxel51.com/fiftyone/workflows/uniqueness/)
-   duplicates: identify and remove duplicate samples in the dataset, using the
    [FiftyOne Brain](https://docs.voxel51.com/api/fiftyone.brain.html?highlight=duplicate#fiftyone.brain.compute_exact_duplicates)
-   similarity: generate a vector similarity index on the dataset, which can be
    used to compare samples in the dataset, using the
    [FiftyOne Brain](https://docs.voxel51.com/user_guide/brain.html#similarity)
-   dimensionality reduction: reduce the dimensionality of feature vectors for
    each sample, using the
    [FiftyOne Brain](https://docs.voxel51.com/user_guide/brain.html#visualizing-embeddings)
    using UMAP, PCA, or t-SNE, so that they can be visualized in 2D or 3D
-   clustering: cluster samples in the dataset using KMeans, DBSCAN, and other
    clustering algorithms, using FiftyOne's
    [Clustering plugin](https://github.com/jacobmarks/clustering-plugin)

Here's some examples of computational queries you can ask VoxelGPT:

-   Compute the brightness of images across my dataset
-   Score the uniqueness of each image in my dataset
-   Generate a similarity index for my dataset
-   Cluster my dataset using KMeans
-   Help me visualize my dataset in 2D using UMAP

💡 If you do not want to allow VoxelGPT to run computations, set the
environment variable:

```shell
export VOXELGPT_ALLOW_COMPUTATIONS=false
```

You can also set the minimum dataset size at which VoxelGPT needs to ask for
permission to run computations:

```shell
export VOXELGPT_APPROVAL_THRESHOLD=1000
```

The default value is 100 samples.

### FiftyOne Library queries

VoxelGPT is not only a pair programmer; it is also an educational tool.
VoxelGPT has access to the entire [FiftyOne docs](https://docs.voxel51.com), as
well as all of the blog posts on the [Voxel51 Blog](https://voxel51.com/blog/),
and transcripts from videos on the
[Voxel51 YouTube channel](https://www.youtube.com/channel/UC9GWqiVDwPdQrW70_v4VtlQ).
It can use all of these resources to answer FiftyOne-related questions.

Here's some examples of documentation queries you can ask VoxelGPT:

-   How do I load a dataset from the FiftyOne Zoo?
-   What does the match() stage do?
-   Can I export my dataset in COCO format?
-   Does FiftyOne have any plugins for active learning?

VoxelGPT will provide links to the most helpful resources across Voxel51's
docs, blog, and YouTube channel. For YouTube videos, the links will point
directly to the most relevant timestamp!

### Querying your workspace

VoxelGPT can answer questions about the environment in which you are running
FiftyOne, including:

-   Other datasets you have downloaded
-   Plugins you have installed, and operators within those plugins
-   Your FiftyOne config
-   Your FiftyOne App config

Here's some examples of workspace queries you can ask VoxelGPT:

-   Do I have any COCO datasets?
-   Do I have any plugins for identifying issues in my data?
-   What is my operator timeout set to?

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

For use with your private Azure deployment, see
[here](#using-azure-openai-deployment)

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

### Using Azure OpenAI deployment

You can use VoxelGPT with your private Azure deployment by setting the
following environment variables:

```shell
export OPENAI_API_TYPE=azure
export AZURE_OPENAI_ENDPOINT=<azure_endpoint>
export AZURE_OPENAI_KEY=<azure_api_key>

export AZURE_OPENAI_GPT35_DEPLOYMENT_NAME=<gpt35-deployment-name>
export AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME=<gpt4o-deployment-name>
export AZURE_OPENAI_TEXT_EMBEDDING_3_LARGE_DEPLOYMENT_NAME=<embedding-deployment-name>

```

If any of the first three environment variables is not set, VoxelGPT will
default to using the OpenAI API. For the last three environment variables, if
any of them is not set of the resource is not found, VoxelGPT will default to
using the OpenAI API for that specific model.

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
you can converse with VoxelGPT via `input()` prompts. Navigate to the directory
where VoxelGPT is located and run:

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

-   OpenAI's
    [GPT-3.5-Turbo](https://platform.openai.com/docs/models/gpt-3-5-turbo) and
    [GPT-4o](https://platform.openai.com/docs/models/gpt-4o) to generate
    textual answers and code
-   OpenAI's
    [text-embedding-3-large model](https://platform.openai.com/docs/guides/embeddings/embedding-models)
    to embed input text prompts
-   [LangChain](https://github.com/hwchase17/langchain) provides the connective
    tissue for the application
-   FiftyOne's [plugin framework](https://docs.voxel51.com/plugins/index.html)
    to provide the interactive panel in the
    [FiftyOne App](https://docs.voxel51.com/user_guide/app.html)

## Limitations

### Media types

VoxelGPT provides limited support for videos, grouped datasets, and 3D media.
Basic filtering, querying, and aggregations will still work, but don't expect
deep insights into 3D data.

### Examples

This implementation is based on a limited set of examples, so it may not
generalize well to all datasets. The more specific your query, the better the
results will be. If you find that the results are not what you expect, please
let us know!

## About FiftyOne

If you've made it this far, we'd greatly appreciate if you'd take a moment to
check out [FiftyOne](https://github.com/voxel51/fiftyone) and give us a star!

FiftyOne is an open source library for building high-quality datasets and
computer vision models. It's the engine that powers this project.

Thanks for visiting! 😊

## Join the Community

If you want join a fast-growing community of engineers, researchers, and
practitioners who love visual AI, join the
[FiftyOne Slack community](https://slack.voxel51.com/)! 🚀🚀🚀
