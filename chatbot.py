from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import gradio as gr
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Free Local Embedding Model (No API key needed)
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Free Groq LLM
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

# Connect to ChromaDB
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Set up retriever
num_results = 5
retriever = vector_store.as_retriever(
    search_kwargs={"k": num_results}
)


def stream_response(message, history):
    # Retrieve relevant chunks
    docs = retriever.invoke(message)

    # Combine chunks into knowledge
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    # Format conversation history
    history_text = "\n".join(
        [f"User: {h[0]}\nAssistant: {h[1]}" for h in history]
    )

    # Build RAG prompt
    rag_prompt = f"""
You are an assistant which answers questions based on knowledge provided to you.
While answering, you don't use your internal knowledge,
but solely the information in the "Knowledge" section.
You don't mention anything to the user about the provided knowledge.
If the answer is not in the knowledge, say:
"I couldn't find that information in the provided documents."

Question:
{message}

Conversation History:
{history_text}

Knowledge:
{knowledge}
"""

    print(rag_prompt)

    # Stream response
    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response.content
        yield partial_message


# Gradio UI
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Ask about the document...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

chatbot.launch()