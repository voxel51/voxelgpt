"""
FiftyOne docs query dispatcher.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import pickle
import re
import uuid

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import TokenTextSplitter
import numpy as np
from scipy.spatial.distance import cosine

# pylint: disable=relative-beyond-top-level
from .utils import (
    get_cache,
    get_embedding_function,
    get_llm,
    stream_retriever,
)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_EMBEDDINGS_FILE = os.path.join(ROOT_DIR, "fiftyone_docs_embeddings.pkl")

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

PATTS_TO_LINKS = {
    "FiftyOne Docs": "https://docs.voxel51.com/",
    "FiftyOne User Guide": "https://docs.voxel51.com/user_guide/index.html",
    "FiftyOne Teams": "https://docs.voxel51.com/teams/index.html",
    "FiftyOne Model Zoo": "https://docs.voxel51.com/user_guide/model_zoo/index.html",
    "FiftyOne Dataset Zoo": "https://docs.voxel51.com/user_guide/dataset_zoo/index.html",
    "FiftyOne Plugins": "https://docs.voxel51.com/plugins/index.html",
    "FiftyOne App": "https://docs.voxel51.com/user_guide/app.html",
    "FiftyOne Brain": "https://docs.voxel51.com/user_guide/brain.html",
}


def _make_api_doc_path(name, docs_dir):
    return os.path.join(docs_dir, "api", f"fiftyone.{name}.html")


def _get_docs_build_dir():
    import fiftyone as fo

    fo_repo_dir = os.path.dirname(os.path.dirname(fo.__file__))
    return os.path.join(fo_repo_dir, "docs", "build", "html")


def _get_url_from_path(path):
    rel_path = "".join(path.split("html")[-2:])
    return f"https://docs.voxel51.com{rel_path}html"


def _generate_docs_embeddings():
    """Generates embeddings for the FiftyOne documentation.

    This is a developer method that only needs to be run once after each
    release. It requires a source install of FiftyOne with the fresh docs
    build.
    """

    all_embeddings_dict = {}

    def add_loader_embeddings(loader, all_embeddings_dict, chunk_size=200):
        documents = loader.load()
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0
        )
        texts = text_splitter.split_documents(documents)

        ids = [str(uuid.uuid1()) for _ in texts]
        contents = [text.page_content for text in texts]
        sources = [
            _get_url_from_path(text.metadata["source"]) for text in texts
        ]
        embeddings = model(contents)

        curr_embeddings_dict = {
            id: {"content": content, "embedding": embedding, "source": source}
            for id, content, embedding, source in zip(
                ids, contents, embeddings, sources
            )
        }

        all_embeddings_dict = {**all_embeddings_dict, **curr_embeddings_dict}
        return all_embeddings_dict

    docs_dir = _get_docs_build_dir()
    model = get_embedding_function()

    ## STANDALONE DOCS
    for filename in STANDALONE_DOCS:
        filepath = os.path.join(docs_dir, filename)
        loader = UnstructuredMarkdownLoader(filepath)
        all_embeddings_dict = add_loader_embeddings(
            loader, all_embeddings_dict
        )

    ## DOCS CATEGORIES
    for doc_type in DOC_TYPES:
        print(f"Generating embeddings for {doc_type}...")
        doc_type_dir = os.path.join(docs_dir, doc_type)
        loader = DirectoryLoader(doc_type_dir, glob="**/*.html")
        all_embeddings_dict = add_loader_embeddings(
            loader, all_embeddings_dict
        )

    ## API DOCS
    for api_doc_path in API_DOC_PATHS:
        doc = _make_api_doc_path(api_doc_path, docs_dir)
        loader = UnstructuredMarkdownLoader(doc)
        all_embeddings_dict = add_loader_embeddings(
            loader, all_embeddings_dict
        )

    with open(DOCS_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(all_embeddings_dict, f)


class FiftyOneDocsRetriever(BaseRetriever):
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
        dists = np.array(
            [cosine(query_embedding, emb) for emb in self.embeddings]
        )

        sorted_ix = np.argsort(dists).astype(int)
        return [self.contents[ix] for ix in sorted_ix[:10]]

    async def aget_relevant_documents(self, query: str):
        raise NotImplementedError


def _create_docs_retriever():
    with open(DOCS_EMBEDDINGS_FILE, "rb") as f:
        embeddings = list(pickle.load(f).values())

    return FiftyOneDocsRetriever(embeddings)


def _create_docs_qa_chain():
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=get_docs_retriever(),
    )


def get_docs_retriever():
    cache = get_cache()
    key = "docs_retriever"
    if key not in cache:
        cache[key] = _create_docs_retriever()
    return cache[key]


def get_docs_qa_chain():
    cache = get_cache()
    key = "docs_qa_chain"
    if key not in cache:
        cache[key] = _create_docs_qa_chain()
    return cache[key]


def _wrap_text(patt):
    link = PATTS_TO_LINKS[patt]
    return f"[{patt}]({link})"


def _format_response(response):
    str_response = response

    md_response = response
    for patt in PATTS_TO_LINKS:
        md_response = re.sub(
            patt, _wrap_text(patt), md_response, flags=re.IGNORECASE
        )

    return {
        "string": str_response,
        "markdown": md_response,
    }


def run_docs_query(query):
    docs_qa = get_docs_qa_chain()
    response = docs_qa({"question": query})
    answer = response["answer"]
    sources = response["sources"].split(",")
    response = answer + "Sources:\n" + "\n".join(sources)
    return _format_response(response)


def stream_docs_query(query):
    retriever = get_docs_retriever()
    for content in stream_retriever(retriever, query):
        if isinstance(content, Exception):
            raise content

        yield content
