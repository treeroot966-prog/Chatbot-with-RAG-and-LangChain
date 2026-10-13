from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Free Embedding Model (No API key needed)
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Initiate vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Load PDF documents
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} pages from PDFs")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)
print(f"Created {len(chunks)} chunks")

# Create unique IDs for each chunk
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Add chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)
print("Documents successfully added to ChromaDB")