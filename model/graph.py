import sqlite3
from typing import List, Annotated
from pydantic import BaseModel
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from chromadb import PersistentClient
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from sentence_transformers import SentenceTransformer
from langgraph.checkpoint.sqlite import SqliteSaver

# Load environment variables
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile")
# Load once at the top of the file — not inside the function
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
# --- State Definitions ---

class RewrittenQuery(BaseModel):
    rewritten_query: str

class MizanAnswer(BaseModel):
    answer: str
    sources: List[str]
    confidence: str  # "high", "medium", "low"
    
class MizanState(TypedDict):
    query: str          # original question
    rewritten_query: str  # improved query after grading fails
    retrieved_docs: List[str]  # chunks from vector DB
    doc_scores: List[float]    # relevance score per chunk
    answer: str         # final output
    sources: List[str]  # where each chunk came from
    attempts: int       # retry counter — safety brake
    language: str 

# --- Node Implementations ---
def detect_language(text: str) -> str:
    # Arabic unicode range
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "ar" if arabic_chars > 0 else "en"

def retrieve_node(state: MizanState):
    ''' Retrieves relevant chunks from ChromaDB based on the query. '''
    query = state.get("rewritten_query") or state["query"]
    language = detect_language(query)

    client = PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="mizan_laws")
    
    # Embed the query with the SAME model used in ingest.py
    query_embedding = embed_model.encode(query).tolist()
    
    # Use query_embeddings not query_texts
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    return {
        "retrieved_docs": results["documents"][0] if results["documents"] else [],
        "doc_scores": results["distances"][0] if results["distances"] else [],
        "sources": results["ids"][0] if results["ids"] else [],
        "language": language
    }
# TODO: parallelize grading using Send API for better performance
# def grader(state: MizanState):

def grader(state: MizanState):
    """
    Grades each retrieved chunk — relevant or not.
    """
    
    # Get what we need from state
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    
    # Grade each chunk individually
    scores = []
    for doc in retrieved_docs:
        prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.
        
Question: {query}

Retrieved chunk: {doc}

Is this chunk relevant to the question? Answer only 'yes' or 'no'."""
        
        result = llm.invoke(prompt)
        score = 1.0 if "yes" in result.content.lower() else 0.0
        scores.append(score)
    
    return {"doc_scores": scores}


def route_after_grading(state: MizanState):
    scores = state.get("doc_scores", [])
    attempts = state.get("attempts", 0)
    
    # If any chunk is relevant OR we've retried too many times → generate
    if any(score >= 0.5 for score in scores) or attempts >= 2:
        return "generate"
    
    # Otherwise rewrite and retry
    return "rewrite"

def rewrite_node(state: MizanState):
    
    query = state["query"]
    attempts = state.get("attempts", 0)
    
    system_prompt = """You are a helpful assistant that rewrites user questions 
to be more specific and effective for searching Egyptian legal documents.
Return a more detailed and precise version of the question."""
    
    structured_llm = llm.with_structured_output(MizanAnswer)

    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Question: {query}

    Context:
    {context}

    Sources: {sources}

    Answer:""")
    ])

    return {
        "answer": result.answer,
        "sources": result.sources
    }

def generate_node(state: MizanState):
    ''' Generates an answer based on the retrieved documents. '''
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    sources = state["sources"]
    language = state.get("language", "ar")  # default to Arabic
    
    # Combine retrieved chunks into one context block
    context = "\n\n".join(retrieved_docs)
    
    system_prompt = f"""You are Mizan, an expert legal assistant specializing in Egyptian law.
Your job is to answer questions about Egyptian laws clearly and accurately.
Use ONLY the provided context to answer. Do not make up information.
Always cite the source of your answer.
Respond in {"Arabic" if language == "ar" else "English"}."""

    result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Question: {query}

Context:
{context}

Sources: {sources}

Answer:""")
    ])
    
    return {"answer": result.content}


# graph definition
mizan_graph = StateGraph(MizanState)
mizan_graph.add_node('retrieve', retrieve_node)
mizan_graph.add_node('grade', grader)
mizan_graph.add_node('rewrite', rewrite_node)
mizan_graph.add_node('generate', generate_node)
mizan_graph.add_edge(START, 'retrieve')
mizan_graph.add_edge('retrieve', 'grade')
mizan_graph.add_conditional_edges("grade", route_after_grading, ["generate", "rewrite"])
mizan_graph.add_edge("rewrite", "retrieve")
mizan_graph.add_edge("generate", END)

conn = sqlite3.connect("./mizan_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = mizan_graph.compile(checkpointer=memory)

result = graph.invoke(
    {"query": "ما هي حقوق العامل في إجازة الأمومة؟", "attempts": 0},
    config={"configurable": {"thread_id": "1"}}
)
print(result["answer"])
print("\n--- MIZAN STREAMING ---")
for chunk in graph.stream(
    {"query": "ما هي حقوق العامل في إجازة الأمومة؟", "attempts": 0},
    config={"configurable": {"thread_id": "3"}},
    stream_mode="updates"
):
    node_name = next(iter(chunk.keys()))
    print(f"\n-- Node: {node_name} --")
    if "answer" in chunk.get(node_name, {}):
        print(chunk[node_name]["answer"])