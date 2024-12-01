from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank


from typing import Literal
from pydantic import BaseModel, Field


# Supervisor Node
def supervisor_agent(state):
    question = state['question']
    groq_llama3_1_70b = state['groq_llama3_1_70b']

    # Data model
    class SupervisorRoute(BaseModel):
        """Route a user query to the most relevant datasource."""

        route: Literal["vector_store_agent", "apology_agent"] = Field(
            description="Given a user question choose to route it to vector_store_agent or apology_agent. ",
        )


    structured_supervisor_llm = groq_llama3_1_70b.with_structured_output(SupervisorRoute)

    # Prompt
    system = """You are an expert at routing a user question to a vector_store or web search.
    The vectorstore contains documents related to machine learning operation, machine learning and data science.
    Use the vector_store_agent for questions on these topics. Otherwise, use apology_agent."""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    supervisor_agent = route_prompt | structured_supervisor_llm

    supervisor_choice = supervisor_agent.invoke(question)

    return {"supervisor_route_choice": supervisor_choice.route}

# Mlops Agent
def mlops_agent(state):
    query = state['question']
    hybrid_rerank_qdrant_retriever = state['hybrid_rerank_qdrant_retriever']
    groq_llama3_1_70b = state['groq_llama3_1_70b']

    # -----------------------------------------------------------------
    # 1. Retrieval
    # -----------------------------------------------------------------

    compressor = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2",
        top_n=4
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_rerank_qdrant_retriever
    )

    retrieved_docs = compression_retriever.invoke(query)

    # Format the docs
    doc_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # -----------------------------------------------------------------
    # 2. Generation
    # -----------------------------------------------------------------
    # Prompt
    template = """
    You are an assistant for question-answering tasks. 
    Please answer the question based on the context provided.
    Do not tell the user that you are referring to the context to answer the question 
    If you don't know the answer or the context does not answer the question, just say that you don't know. 
    Elaborate your answer in well structured format.

    Question: {question} 
    Context: {context} 
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)


    llm = groq_llama3_1_70b

    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    rag_answer =  rag_chain.invoke({"question": query, "context":doc_context})

    return {'rag_answer': rag_answer}

# Apology Agent
def apology_agent(state):
    question = state['question']
    groq_llama3_1_70b = state['groq_llama3_1_70b']

    # Prompt
    template = """
    You are an assistant for question-answering tasks related to machine learning operation, machine learning and data science only. 
    The question are not related to machine learning operation, machine learning and data science.
    Do not answer the question and apologize to the user.

    Question: {question} 
    Apology:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = groq_llama3_1_70b

    apology_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    apology_text =  apology_chain.invoke({"question": question})

    return {
        "rag_answer": apology_text
    }