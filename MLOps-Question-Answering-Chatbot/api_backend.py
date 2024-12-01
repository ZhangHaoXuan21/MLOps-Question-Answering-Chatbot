from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
import os

from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

from mlops_agents.agent import MlOpsAgent

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5"
)

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY,
    collection_name="mlops_document",
    retrieval_mode=RetrievalMode.HYBRID,
)

hybrid_rerank_qdrant_retriever = qdrant.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 20},
)

groq_llama3_1_70b = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)

app = FastAPI()

# Define the Pydantic model
class QueryState(BaseModel):
    user_query: str


@app.post("/chatResponse")
async def chat_response(request: QueryState):
    # Access the incoming JSON
    user_query = request.user_query

    graph_state = {
        "question":user_query,
        "rag_answer":"None",
        "supervisor_route_choice":"None",
        "hybrid_rerank_qdrant_retriever":hybrid_rerank_qdrant_retriever,
        "groq_llama3_1_70b": groq_llama3_1_70b
    }

    mlops_agent = MlOpsAgent()

    result = mlops_agent(graph_state)
    
    # Return a JSON response
    return {
        "response": result['rag_answer'],
    }
