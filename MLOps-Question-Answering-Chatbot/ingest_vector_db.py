from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import config



def ingest_to_vector_db():
    # -------------------------------------------------------------------------
    # 1. Load multiple PDFs using PyPDFLoader, only if the file is a PDF
    # -------------------------------------------------------------------------
    print("Loading PDFs Start.")

    file_paths = config.PDF_FILE_PATHS
    docs = []

    for file_path in file_paths:
        # Check if the file is a PDF
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())  # Combine all loaded docs into a single list
        else:
            print(f"Skipping non-PDF file: {file_path}")

    print("Loading PDFs End.")

    # Preprocess page content
    for doc in docs:
        # Replace tabs with spaces in each document's page_content
        doc.page_content = doc.page_content.replace("\t", " ")

    # -------------------------------------------------------------------------
    # 2. Ingest to Vector Database: Qdrant
    # -------------------------------------------------------------------------
    print("Ingest to Vector Database Start.")
    url = config.QDRANT_URL
    api_key = config.QDRANT_API_KEY

    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm25"
    )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:v1.5"
    )

    
    QdrantVectorStore.from_documents(
        docs,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=url,
        prefer_grpc=True,
        api_key=api_key,
        collection_name="mlops_document",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    print("Ingest to Vector Database End.")


if __name__ == "__main__":
    ingest_to_vector_db()



















