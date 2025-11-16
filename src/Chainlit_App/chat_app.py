import chainlit as cl

from src.Part1_Simple_LLM.simple_llm_chat import run_chat as simple_llm_chat
from src.Part2_RAG.rag_chat import run_chat as rag_chat
from src.Part3_GraphRAG.graphrag_chat import run_chat as graph_rag_chat
from src.Part4_Text2SQL.text_to_sql_chat import (
    write_sql_query,
    run_sql_query,
    write_answer,
    write_plotly_figure,
)
from src.Part5_Agent.frc_agent import run_frc_agent

# Instrument the OpenAI client
cl.instrument_openai()


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Simple LLM Chat",
            markdown_description="Simple LLM Chat Assistant specialized in providing information about the Dabburiya Tech community.",
        ),
        cl.ChatProfile(
            name="RAG",
            markdown_description="RAG Assistant specialized in providing information about the Dabburiya Tech members.",
        ),
        cl.ChatProfile(
            name="GraphRAG",
            markdown_description="GraphRAG Assistant specialized in providing information about the Dabburiya Tech members.",
        ),
        cl.ChatProfile(
            name="Text-to-SQL",
            markdown_description="Text-to-SQL Assistant specialized in providing information about the Dabburiya Tech members stats.",
        ),
        cl.ChatProfile(
            name="FRC Agent",
            markdown_description="FRC Agent specialized in providing information about the FRC teams.",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"starting chat using the {chat_profile} chat profile",
    ).send()


@cl.step(type="Simple LLM Chat")
async def run_simple_llm_chat(question: str) -> str:
    final_answer = simple_llm_chat(question)
    await cl.Message(content=final_answer).send()


@cl.step(type="RAG Chat")
async def run_rag_chat(question: str) -> str:
    final_answer = rag_chat(question)
    await cl.Message(content=final_answer).send()


@cl.step(type="GraphRAG Chat")
async def run_graphrag_chat(question: str) -> str:
    final_answer = graph_rag_chat(question)
    await cl.Message(content=final_answer).send()


@cl.step(type="Text-to-SQL Chat")
async def run_text_to_sql_chat(question: str) -> str:

    # Show SQL query in collapsed section
    async with cl.Step(name="Generating SQL Query") as step:
        # write SQL query
        sql_query_result = write_sql_query(question)

        step.output = sql_query_result.model_dump_json(indent=2)
        step.language = "json"
        step.update()

    # Show SQL query in collapsed section
    async with cl.Step(name="Running SQL Query") as step:
        # run SQL query
        query_results = run_sql_query(sql_query_result.sql_query)

        step.output = query_results.to_string()
        step.update()

    # Generate final answer
    final_answer = write_answer(question, query_results.to_string())
    fig = write_plotly_figure(question, query_results.to_string())
    if fig:
        elements = [cl.Plotly(name="chart", figure=fig, display="inline")]
    else:
        elements = []
    await cl.Message(content=final_answer, elements=elements).send()


@cl.step(type="run_frc_agent")
async def run_frc_agent_chat(question: str) -> str:
    result = await run_frc_agent(question)
    await cl.Message(content=result.final_output).send()


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Simple LLM Chat":
        await run_simple_llm_chat(message.content)
    elif chat_profile == "RAG":
        await run_rag_chat(message.content)
    elif chat_profile == "GraphRAG":
        await run_graphrag_chat(message.content)
    elif chat_profile == "Text-to-SQL":
        await run_text_to_sql_chat(message.content)
    elif chat_profile == "FRC Agent":
        await run_frc_agent_chat(message.content)
