from glob import glob
import json
import os
import uuid

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_embedding_function

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_EMBEDDINGS_DIR = os.path.join(ROOT_DIR, ".fiftyone_docs_embeddings")

CHROMADB_DIR = '.fiftyone_docs_db'

FIFTYONE_DIR = os.getenv("FIFTYONE_DIR")
FIFTYONE_DOCS_DIR = os.path.join(
    FIFTYONE_DIR, 
    "docs",
    "build",
    "html"
    )

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

def _generate_embeddings():
    """
    Only run once. Generates embeddings for all the docs in the fiftyone docs.
    Requires the fiftyone docs to be cloned locally, and that 
    `bash docs/generate_docs.bash` has been run.
    """

    for doc_type in DOC_TYPES:
        print(f"Generating embeddings for {doc_type}...")
        doc_type_dir = os.path.join(FIFTYONE_DOCS_DIR, doc_type)
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
            id: {
            "content": content,
            "embedding": embedding
            }
            for id, content, embedding in zip(ids, contents, embeddings)
        }

        with open(doc_type_embeddings_file, "w") as f:
            json.dump(embeddings_dict, f)

def _create_vectorstore():
    """
    Creates the vectorstore for the fiftyone docs.
    """
    docs_db = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMADB_DIR,
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

def load_vectorstore():
    """
    Loads the vectorstore for the fiftyone docs.
    """
    docs_db = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMADB_DIR,
        )

    return docs_db

def run_docs_query(query):
    """
    Queries the fiftyone docs for the given query.
    """
    docs_db = load_vectorstore()
    docs_qa = RetrievalQA.from_chain_type(
        llm=get_llm(), 
        chain_type="stuff", 
        retriever=docs_db.as_retriever()
        )

    return docs_qa.run(query)

