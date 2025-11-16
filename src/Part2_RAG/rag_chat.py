import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

# Load environment variables from .env file
load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with GitHub Models
client = OpenAI(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
)

# Initialize Azure OpenAI Embedding model
embedding_client = OpenAIEmbedding(
    api_base=OPENAI_API_BASE, api_key=OPENAI_API_KEY, model="text-embedding-3-small"
)


def load_index():
    source_dir = Path(__file__).resolve().parent.parent.parent

    # Load vectors from existing LanceDB vector store
    vector_directory_path = source_dir / "data" / "lancedb"
    vector_store = LanceDBVectorStore(
        uri=f"{vector_directory_path}", query_type="vector"
    )

    # Create storage context with the vector store
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embedding_client
    )

    # Create vector retriever from the vector index
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)
    return vector_retriever


def run_llm_response(question: str, retrieved_context: str) -> str:
    DEVELOPER_PROMPT = """
    # Identity
    You are a helpful assistant that provides information only about the Dabburiya Tech community members based on the provided context.

    # Instructions
    The assistant answers only questions related to the purpose of this chat. If the user asks something outside the chat purpose, reply:"The question is outside the purpose of this chat."
    If the question is related to the chat purpose but the answer is not found in the provided CONTEXT, reply: "Not in provided context."
    The assistant never invents facts not present in the CONTEXT.

    # Examples
    """

    USER_PROMPT_TEMPLATE = """
    You are being asked a question about the Dabburiya Tech community members.

    QUESTION:
    {question}

    CONTEXT:
    {retrieved_context}

    INSTRUCTIONS:
    Answer the QUESTION using only the CONTEXT.
    If the answer is not found in the CONTEXT, reply: "Not in provided context".
    """

    USER_PROMPT = USER_PROMPT_TEMPLATE.format(
        retrieved_context=retrieved_context, question=question
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    return response.choices[0].message.content


def run_chat(question: str) -> str:

    # Load vector retriever
    vector_retriever = load_index()

    # Retrieve relevant context from the vector store
    retrieved_context = vector_retriever.retrieve(question)
    retrieved_context_str = ""
    for node in retrieved_context:
        retrieved_context_str += "\n" + node.get_content()

    response = run_llm_response(question, retrieved_context_str)
    return response


if __name__ == "__main__":
    question = "where nadeem azaizah currently working?"
    print(f"\n🔍 User question: {question}")

    # Get response from the LLM based on RAG retrieved context
    response = run_chat(question)
    print(f"💡 Chat Response:\n{response}")
