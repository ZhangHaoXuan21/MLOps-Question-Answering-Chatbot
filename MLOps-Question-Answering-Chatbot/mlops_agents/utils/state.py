from langgraph.graph import MessagesState
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq


class GraphState(MessagesState):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        supervisor_route_choice: Route Choice of supervisor
        hybrid_rerank_qdrant_retriever: Qdrant Vector Store Retriever
        groq_llama3_1_70b: Llama3.1 70b from GROQ API
    """

    question: str
    rag_answer: str
    supervisor_route_choice: str
    hybrid_rerank_qdrant_retriever: QdrantVectorStore
    groq_llama3_1_70b: ChatGroq