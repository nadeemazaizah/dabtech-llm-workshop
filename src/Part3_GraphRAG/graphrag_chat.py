import os
from pathlib import Path
import asyncio
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Load environment variables from .env file
load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

setup_logger("lightrag", level="INFO")

source_dir = Path(__file__).resolve().parent.parent.parent
WORKING_DIR = os.path.join(source_dir, "data/lightrag_storage/")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        max_parallel_insert=4,
        embedding_func=openai_embed,
        llm_model_func=openai_complete,
        cosine_threshold=0.5,
        cosine_better_than_threshold=0.5,
        addon_params={
            "entity_types": [
                "person",
                "company",
                "position",
                "education_institution",
                "study",
                "degree",
                "skill",
                "technology",
                "programming_language",
                "location",
            ]
        },
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag


def run_chat(query):
    graphrag = asyncio.run(initialize_rag())

    query_param = QueryParam(mode="hybrid", response_type="Single Paragraph", top_k=3)

    response = asyncio.run(
        graphrag.aquery(
            query=query,
            param=query_param,
        )
    )
    return response


if __name__ == "__main__":
    question = "where nadeem azaizah currently working?"
    print(f"\n🔍 User question: {question}")

    # Get response from the LLM based on Graph RAG retrieved context
    response = run_chat(question)
    print(f"💡 Chat Response:\n{response}")
    print(response)
