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

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import TokenTextSplitter
import numpy as np
from scipy.spatial.distance import cosine

# pylint: disable=relative-beyond-top-level
from .utils import (
    get_cache,
    get_embedding_function,
    get_llm,
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


def _get_docs_build_dir():
    import fiftyone as fo

    fo_repo_dir = os.path.dirname(os.path.dirname(fo.__file__))
    return os.path.join(fo_repo_dir, "docs", "build", "html")


def _generate_docs_embeddings():
    """Generates embeddings for the FiftyOne documentation.

    This is a developer method that only needs to be run once after each
    release. It requires a source install of FiftyOne with the fresh docs
    build.
    """
    docs_dir = _get_docs_build_dir()
    all_embeddings_dict = {}

    model = get_embedding_function()

    for doc_type in DOC_TYPES:
        print(f"Generating embeddings for {doc_type}...")
        doc_type_dir = os.path.join(docs_dir, doc_type)

        loader = DirectoryLoader(doc_type_dir, glob="**/*.html")
        documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        ids = [str(uuid.uuid1()) for _ in texts]
        contents = [text.page_content for text in texts]
        embeddings = model(contents)

        curr_embeddings_dict = {
            id: {"content": content, "embedding": embedding}
            for id, content, embedding in zip(ids, contents, embeddings)
        }

        all_embeddings_dict = {**all_embeddings_dict, **curr_embeddings_dict}

    with open(DOCS_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(all_embeddings_dict, f)


class FiftyOneDocsRetriever(BaseRetriever):
    def __init__(self, embeddings):
        self.contents = [
            Document(page_content=doc["content"]) for doc in embeddings
        ]
        self.embeddings = [np.array(doc["embedding"]) for doc in embeddings]
        self.model = get_embedding_function()

    def get_relevant_documents(self, query):
        query_embedding = np.array(self.model(query))
        dists = np.array(
            [cosine(query_embedding, emb) for emb in self.embeddings]
        )

        sorted_ix = np.argsort(dists).astype(int)
        return [self.contents[ix] for ix in sorted_ix[:5]]

    async def aget_relevant_documents(self, query: str):
        raise NotImplementedError


def _create_docs_qa_chain():
    with open(DOCS_EMBEDDINGS_FILE, "rb") as f:
        embeddings = list(pickle.load(f).values())

    retriever = FiftyOneDocsRetriever(embeddings)
    return RetrievalQA.from_chain_type(
        llm=get_llm(), chain_type="stuff", retriever=retriever
    )


def load_docs_qa_chain():
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


def run_docs_query(query, streaming=False):
    docs_qa = load_docs_qa_chain()
    response = docs_qa.run([query]).strip()
    return _format_response(response)
