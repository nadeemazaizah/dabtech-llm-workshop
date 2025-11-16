import os
from pathlib import Path
import asyncio
import json
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


def load_profiles(profiles_folder):
    text_inputs = []
    documents_ids = []
    profile_files = os.listdir(profiles_folder)
    for profile_file in profile_files:
        with open(
            os.path.join(profiles_folder, profile_file), "r", encoding="utf-8"
        ) as file:
            profile_content = file.read()
        text_inputs.append(str(profile_content))
        documents_ids.append(profile_file)

    print(f"Total profiles to insert: {len(text_inputs)}")
    print(f"Total profiles to insert: {len(documents_ids)}")
    return text_inputs, documents_ids


def index_graphrag():
    graphrag = asyncio.run(initialize_rag())

    profiles_folder = os.path.join(source_dir, "data/profiles_examples/")
    text_inputs, documents_ids = load_profiles(profiles_folder)

    asyncio.run(graphrag.ainsert(input=text_inputs, ids=documents_ids))
    asyncio.run(graphrag.finalize_storages())
    return graphrag


if __name__ == "__main__":
    graphrag = index_graphrag()

    query = "where nadeem azaizah currently working?"

    query_param = QueryParam(
        mode="hybrid",
        top_k=3,
    )

    final_data = asyncio.run(graphrag.aquery_data(query=query, param=query_param))

    print("Final retrieved data:")
    print(final_data)
