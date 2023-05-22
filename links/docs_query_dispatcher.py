"""
FiftyOne docs query dispatcher.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from glob import glob
import json
import os
import re
import uuid

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_embedding_function, get_cache


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_EMBEDDINGS_DIR = os.path.join(ROOT_DIR, ".fiftyone_docs_embeddings")
CHROMADB_DOCS_DIR = os.path.join(ROOT_DIR, ".fiftyone_docs_db")

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
    "FiftyOne Brain": "https://docs.voxel51.com/user_guide/brain.html"
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
    for doc_type in DOC_TYPES:
        print(f"Generating embeddings for {doc_type}...")
        doc_type_dir = os.path.join(docs_dir, doc_type)
        doc_type_embeddings_file = os.path.join(
            DOCS_EMBEDDINGS_DIR,
            doc_type + "_embeddings.json",
        )

        loader = DirectoryLoader(doc_type_dir, glob="**/*.html")
        documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        ids = [str(uuid.uuid1()) for _ in texts]
        contents = [text.page_content for text in texts]
        embeddings = get_embedding_function()(contents)

        embeddings_dict = {
            id: {"content": content, "embedding": embedding}
            for id, content, embedding in zip(ids, contents, embeddings)
        }

        with open(doc_type_embeddings_file, "w") as f:
            json.dump(embeddings_dict, f)


def _create_docs_vectorstore():
    docs_db = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMADB_DOCS_DIR,
    )
    docs_db.persist()

    docs_embeddings_files = glob(os.path.join(DOCS_EMBEDDINGS_DIR, "*.json"))
    for docs_embeddings_file in docs_embeddings_files:
        ids = []
        embeddings = []
        documents = []

        with open(docs_embeddings_file, "r") as f:
            docs_embeddings = json.load(f)

        for doc_id, doc in docs_embeddings.items():
            ids.append(doc_id)
            embeddings.append(doc["embedding"])
            documents.append(doc["content"])

        docs_db._collection.add(
            metadatas=None,
            embeddings=embeddings,
            documents=documents,
            ids=ids,
        )


def _load_docs_vectorstore():
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMADB_DOCS_DIR,
    )


def initialize_docs_qa_chain():
    cache = get_cache()
    docs_db = _load_docs_vectorstore()
    docs_qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(), chain_type="stuff", retriever=docs_db.as_retriever()
    )
    cache["docs_qa_chain"] = docs_qa_chain


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
    cache = get_cache()
    if "docs_qa_chain" not in cache:
        initialize_docs_qa_chain()
    docs_qa = cache["docs_qa_chain"]
    response = docs_qa.run(query).strip()
    return _format_response(response)
