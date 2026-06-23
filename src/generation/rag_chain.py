from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL
from src.retrieval.retriever import trs_retriever


# format
def format_doc(docs):
    if not docs:
        return "No relevent data found in the knowledge base"
    formatted = []
    for doc in docs:
        timestamp = doc.metadata.get("timestamp", "unknown")
        source_type = doc.metadata.get("source_type", "unknown")
        credibility = doc.metadata.get("credibility", "unknown")
        text = f"[Source: {source_type} | Credibility: {credibility} | Retrieved at: {timestamp}]\n{doc.page_content}"
        formatted.append(text)
    return "\n\n".join(formatted)


# prompt template
prompt = ChatPromptTemplate.from_template("""
          you are a smart city assistant
          Each context block starts with [Source | Credibility | Retrieved at: TIMESTAMP].
          This metadata is for internal use only. Do NOT include it in your answer.           
          The retrieved at timestamp is the ONLY trusted time reference in the system.                                                            
      
          context:
          {context}

          question:
          {question}

          Answer:
""")


# RAG chain
def build_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    rag_chain = (
        {
            "context": lambda q: format_doc(trs_retriever(vectorstore, q)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
