import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.retrievers import VectorIndexRetriever

# Load environment variables from .env file
load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Azure OpenAI Embedding model
embedding_client = OpenAIEmbedding(
    api_base=OPENAI_API_BASE, api_key=OPENAI_API_KEY, model="text-embedding-3-small"
)


def index_documents():
    source_dir = Path(__file__).resolve().parent.parent.parent

    # Load documents from the specified directory
    data_directory_path = source_dir / "data" / "profiles_examples"
    documents = SimpleDirectoryReader(data_directory_path).load_data()
    print(len(documents))

    vector_directory_path = source_dir / "data" / "lancedb"
    vector_store = LanceDBVectorStore(
        uri=f"{vector_directory_path}", mode="overwrite", query_type="vector"
    )

    # Create storage context with the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create vector index from documents
    vector_index = VectorStoreIndex.from_documents(
        documents=documents,
        show_progress=True,
        storage_context=storage_context,
        embed_model=embedding_client,
    )

    # Create vector retriever from the vector index
    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3,
    )
    return vector_retriever


if __name__ == "__main__":

    vector_retriever = index_documents()

    question = "where nadeem azaizah currently working?"
    print(f"\n🔍 User question: {question}")

    vector_retrieved_nodes = vector_retriever.retrieve(question)

    retrieved_context = ""
    for node in vector_retrieved_nodes:
        retrieved_context += "\n" + node.text

    print(f"💡 Retrieved Context:\n{retrieved_context}")
