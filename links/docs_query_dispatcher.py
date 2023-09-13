"""
FiftyOne docs query dispatcher.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from glob import glob
import numpy as np
import os
import pickle
import uuid

from langchain.schema import Document, BaseRetriever
import numpy as np
from scipy.spatial.distance import cosine

# pylint: disable=relative-beyond-top-level
from .utils import (
    count_tokens,
    get_cache,
    get_embedding_function,
    stream_retriever,
    query_retriever,
)

from .markdown_utils import get_markdown_documents


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

DOCS_EMBEDDINGS_FILE = os.path.join(ROOT_DIR, "fiftyone_docs_embeddings.pkl")
API_DOCS_EMBEDDINGS_FILE = os.path.join(
    ROOT_DIR, "fiftyone_api_embeddings.pkl"
)

PROMPT_TEMPLATE_FILE = os.path.join(PROMPTS_DIR, "docs_qa_template.txt")

DOC_TYPES = (
    "cheat_sheets",
    "cli",
    "environments",
    "faq",
    "getting_started",
    "integrations",
    "plugins",
    "recipes",
    "teams",
    "tutorials",
    "user_guide",
)

API_DOC_PATHS = (
    "brain",
    "core.plots",
    "core.session",
    "core.aggregations",
    "core.annotation",
    "core.brain",
    "core.collections",
    "core.evaluation",
    "core.expressions",
    "core.frame",
    "core.labels",
    "core.models",
    "core.sample",
    "core.spaces",
    "core.stages",
    "core.view",
    "types",
)

STANDALONE_DOCS = (
    "index.html",
    "release-notes.html",
)

BAD_PATTERNS = (
    "ts-api",
    "detection_mistakenness",
    "model_inference",
    "label_mistakes",
    "dataset_creation/zoo",
    "dataset_creation/common_datasets",
)


def _make_api_doc_path(name, docs_dir):
    return os.path.join(docs_dir, "api", f"fiftyone.{name}.html")


def _get_docs_build_dir():
    import fiftyone as fo

    fo_repo_dir = os.path.dirname(os.path.dirname(fo.__file__))
    return os.path.join(fo_repo_dir, "docs", "build", "html")


def _get_url(path, anchor):
    rel_path = "".join(path.split("html")[-2:])
    page_url = f"https://docs.voxel51.com{rel_path}html"
    if anchor:
        anchor = ".".join(anchor.split(".")[:4])
        return page_url + "#" + anchor
    else:
        return page_url


def _generate_file_embeddings(filepath):
    model = get_embedding_function()
    md_docs_dict = get_markdown_documents(filepath)

    ids = []
    contents = []
    sources = []
    for anchor, section in md_docs_dict.items():
        ids.append(str(uuid.uuid1()))
        source = _get_url(filepath, anchor)
        for chunk in section:
            contents.append(chunk.page_content)
            sources.append(source)

    embeddings = model(contents)
    curr_embeddings_dict = {
        id: {"content": content, "embedding": embedding, "source": source}
        for id, content, embedding, source in zip(
            ids, contents, embeddings, sources
        )
    }

    return curr_embeddings_dict


def _get_docs_path_list():
    docs_dir = _get_docs_build_dir()

    doc_paths = []

    ### add standalone docs
    standalone_paths = [
        os.path.join(docs_dir, filename) for filename in STANDALONE_DOCS
    ]
    doc_paths.extend(standalone_paths)

    ### add remaining types of docs
    for doc_type in DOC_TYPES:
        doc_type_dir = os.path.join(docs_dir, doc_type)
        doc_paths.extend(
            glob(os.path.join(doc_type_dir, "*.html"), recursive=True)
        )
        doc_paths.extend(
            glob(os.path.join(doc_type_dir, "*/*.html"), recursive=True)
        )

    good_doc_paths = []
    for doc_path in doc_paths:
        if any([bad_patt in doc_path for bad_patt in BAD_PATTERNS]):
            continue
        else:
            good_doc_paths.append(doc_path)

    ### add api docs
    api_paths = [
        _make_api_doc_path(api_doc_path, docs_dir)
        for api_doc_path in API_DOC_PATHS
    ]
    # doc_paths.extend(api_paths)

    return api_paths, good_doc_paths


def _generate_docs_embeddings():
    """Generates embeddings for the FiftyOne documentation.

    This is a developer method that only needs to be run once after each
    release. It requires a source install of FiftyOne with the fresh docs
    build.
    """

    all_embeddings_dict = {}

    api_paths, doc_paths = _get_docs_path_list()

    for doc_path in doc_paths:
        print(doc_path)
        curr_embeddings_dict = _generate_file_embeddings(doc_path)
        all_embeddings_dict.update(curr_embeddings_dict)

    with open(DOCS_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(all_embeddings_dict, f)

    api_embeddings_dict = {}
    for api_path in api_paths:
        print(api_path)
        curr_embeddings_dict = _generate_file_embeddings(api_path)
        api_embeddings_dict.update(curr_embeddings_dict)

    with open(API_DOCS_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(api_embeddings_dict, f)


class FiftyOneDocsRetriever:
    def __init__(self, embeddings):
        self.model = get_embedding_function()
        self.contents = [
            Document(
                page_content=doc["content"],
                metadata={"source": doc["source"]},
            )
            for doc in embeddings
        ]
        self.embeddings = [np.array(doc["embedding"]) for doc in embeddings]

    def get_relevant_documents(self, query):
        query_embedding = np.array(self.model(query))
        if len(query_embedding.shape) == 2:
            query_embedding = query_embedding[0]
        dists = np.array(
            [cosine(query_embedding, emb) for emb in self.embeddings]
        )

        sorted_ix = np.argsort(dists).astype(int)
        relevant_docs = [self.contents[ix] for ix in sorted_ix[:10]]
        lens = [count_tokens(doc.page_content) for doc in relevant_docs]
        cumsums = np.cumsum(lens)
        cutoff = 3200
        if cumsums[-1] > cutoff:
            relevant_docs = [
                doc for doc, cs in zip(relevant_docs, cumsums) if cs < cutoff
            ]
        return relevant_docs

    async def aget_relevant_documents(self, query: str):
        raise NotImplementedError


def get_prompt_template():
    cache = get_cache()
    key = "docs_prompt_template"
    if key not in cache:
        cache[key] = _load_prompt_template()
    return cache[key]


def _load_prompt_template():
    with open(PROMPT_TEMPLATE_FILE, "r") as f:
        return f.read()


def _create_docs_retriever():
    with open(DOCS_EMBEDDINGS_FILE, "rb") as f:
        embeddings = list(pickle.load(f).values())

    with open(API_DOCS_EMBEDDINGS_FILE, "rb") as f:
        embeddings.extend(list(pickle.load(f).values()))

    return FiftyOneDocsRetriever(embeddings)


def get_docs_retriever():
    cache = get_cache()
    key = "docs_retriever"
    if key not in cache:
        cache[key] = _create_docs_retriever()
    return cache[key]


def run_docs_query(query):
    retriever = get_docs_retriever()
    prompt_template = get_prompt_template()
    return query_retriever(retriever, prompt_template, query)


def stream_docs_query(query):
    retriever = get_docs_retriever()
    prompt_template = get_prompt_template()
    for content in stream_retriever(retriever, prompt_template, query):
        if isinstance(content, Exception):
            raise content

        yield content
