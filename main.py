import os
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# ==========================================
# 1. State Definition
# ==========================================
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    intent: str
    needs_human: bool

# ==========================================
# 2. Document Processing & RAG Setup
# ==========================================
DB_DIR = "./chroma_db"
DATA_DIR = "./data"

def initialize_vector_store():
    # Using free local HuggingFace embeddings for the vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if we already have a persistent DB
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        print("Loading existing ChromaDB...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    print("Initializing new ChromaDB from Insurance PDF...")
    # Find a PDF in the data directory
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError("Please place the Bajaj Insurance PDF in the 'data/' directory.")
    
    file_path = os.path.join(DATA_DIR, pdf_files[0])
    
    # Load and Chunk the PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    # Store Embeddings in ChromaDB
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    return vectorstore

# Initialize global components
vectorstore = initialize_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Groq LLM (Using the newer Llama 3.1 model)
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================================
# 3. Graph Nodes
# ==========================================
def retrieve_node(state: GraphState):
    """Retrieves context from the vector database."""
    print("---NODE: RETRIEVE---")
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    # If no context is found, flag for HITL
    if not context.strip():
        return {"context": "No context found.", "needs_human": True}
        
    return {"context": context, "needs_human": False}

def evaluate_intent_node(state: GraphState):
    """Evaluates if the query requires human escalation based on context."""
    print("---NODE: EVALUATE INTENT---")
    if state.get("needs_human"):
        return {"intent": "escalate"}
        
    # Ask LLM if the context is sufficient to answer the question
    prompt = f"""
    Given the following context from a Bajaj General Insurance policy, can you accurately answer the user's question?
    Context: {state['context']}
    Question: {state['question']}
    
    Respond strictly with 'yes' or 'no'.
    """
    response = llm.invoke(prompt)
    
    # If the LLM says 'no', escalate to a human
    if "no" in response.content.lower():
        return {"intent": "escalate", "needs_human": True}
    return {"intent": "answer", "needs_human": False}

def generate_answer_node(state: GraphState):
    """Generates the final answer using the LLM."""
    print("---NODE: GENERATE ANSWER---")
    # Insurance-specific persona added to the prompt
    prompt = PromptTemplate(
        template="You are a helpful customer support assistant for Bajaj General Insurance. Answer the customer's question based strictly on the policy document context below. Do not make up any insurance rules. If you do not know the answer based on the context, say so.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )
    chain = prompt | llm
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"answer": response.content}

def hitl_node(state: GraphState):
    """Human-in-the-Loop Escalation."""
    print("---NODE: HITL ESCALATION---")
    print(f"\n[SYSTEM ALERT]: The AI could not confidently answer the following question using the Bajaj policy.")
    print(f"User Question: {state['question']}")
    human_response = input("Human Agent, please provide an answer (or type 'cancel'): ")
    return {"answer": f"[Agent Assisted]: {human_response}"}

# ==========================================
# 4. Conditional Routing
# ==========================================
def route_decision(state: GraphState):
    """Routes to either generate answer or human escalation based on intent."""
    if state["intent"] == "escalate":
        return "human_escalation"
    return "generate_answer"

# ==========================================
# 5. Build LangGraph Workflow
# ==========================================
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate_intent", evaluate_intent_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("human_escalation", hitl_node)

# Add edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluate_intent")
workflow.add_conditional_edges("evaluate_intent", route_decision)
workflow.add_edge("generate_answer", END)
workflow.add_edge("human_escalation", END)

# Compile graph
app = workflow.compile()

# ==========================================
# 6. Main Execution CLI
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Bajaj Insurance RAG Assistant Initialized")
    print("="*50 + "\n")
    
    while True:
        user_query = input("\nCustomer: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        initial_state = {"question": user_query, "context": "", "answer": "", "intent": "", "needs_human": False}
        
        # Execute Graph
        result = app.invoke(initial_state)
        
        print(f"\nAssistant: {result['answer']}")
