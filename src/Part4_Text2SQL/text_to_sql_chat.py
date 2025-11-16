import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import sqlite3
import pandas as pd
import plotly.graph_objects as go

# Load environment variables from .env file
load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with GitHub Models
client = OpenAI(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
)


class SQLQueryOutput(BaseModel):
    sql_query: str = Field(description="The generated SQL query.")
    explanation: str = Field(description="An optional explanation of the query.")


# Function to run chat completion
def write_sql_query(question: str) -> str:
    SYSTEM_PROMPT = """
        You are a helpful assistant that creates SQL queries based on the user question and database schema provided.
        """

    DB_SCHEMA_CONTEXT = """
        DATABASE SCHEMA CONTEXT

        Tables and Columns:

        1. members
        - name (TEXT)
        - current_status (TEXT)
        - current_title (TEXT)
        - current_company (TEXT)
        - first_job_start (TEXT)
        - total_years_experience (FLOAT)
        - years_experience_bucket (TEXT)
        - highest_degree (TEXT)
        - institution (TEXT)
        - graduation_year (FLOAT)
        - linkedin_url (TEXT)
        """

    CREATE_QUERY_PROMPT_TEMPLATE = """
        Given the following input {question}, create a syntactically correct sqlite query to help find the answer. 
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results. 
        You can order the results by a relevant column to return the most meaningful or interesting examples in the database.

        Never query for all the columns from a table; only select the relevant columns needed to answer the question.

        Use only the column names and table relationships explicitly provided below. 
        Be careful not to reference non-existent columns or incorrect table relationships. 
        Pay attention to which column belongs to which table and how tables are linked via their relationships.

        when the question asking about how many, number or count, use COUNT function in SQL.

        Only use the following tables and relationships:
        {database_schema_context}
        """

    USER_PROMPT = CREATE_QUERY_PROMPT_TEMPLATE.format(
        question=question, database_schema_context=DB_SCHEMA_CONTEXT
    )
    try:
        response = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            response_format=SQLQueryOutput,
        )

        return response.choices[0].message.parsed

    except Exception as e:
        return f"Error: {str(e)}"


def run_sql_query(query: str):

    # Connect to the SQLite database
    # Get the directory where this script is located
    source_dir = Path(__file__).resolve().parent.parent.parent
    db_path = os.path.join(source_dir, "data/data.db")
    conn = sqlite3.connect(db_path)

    try:
        # Execute the query and fetch results into a DataFrame
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        return f"Error executing query: {str(e)}"
    finally:
        conn.close()


def write_answer(question: str, context: str) -> str:

    SYSTEM_PROMPT = """
        You are a helpful assistant that provides information about the Dabburiya Tech community based on the provided context.
        """

    USER_PROMPT_TEMPLATE = """
            You are being asked a question about the Dabburiya Tech community.

            Context: {context}

            Question: {question}

            Based on this, provide a helpful and relevant answer focused on general community information.
            """

    USER_PROMPT = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


def write_plotly_figure(question: str, context: str) -> str:
    def extract_code(content: str) -> str:
        # Removes the ```python ... ``` wrapper
        if content.startswith("```"):
            content = content.strip("`")  # remove ```
            content = content.split("\n", 1)[1]  # drop 'python' line
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
        return content

    SYSTEM_PROMPT = """
        You are a helpful assistant that return Python code that creates a Plotly graph_objects chart based on the provided context.
        The context contains data in tabular format.
        """

    USER_PROMPT_TEMPLATE = """
            You are being asked to create a Plotly chart based on the provided context.

            Context: {context}

            Question: {question}

            Based on this, provide Python code that uses Plotly graph_objects to create a relevant chart based on the context and question.
            return only the code needed to create the figure object named 'fig' without any additional explanations or text. 
            Directly start with 'import plotly.graph_objects as go' and end with 'fig' object creation, without any extra text before or after.
            """

    USER_PROMPT = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )

        response_content = response.choices[0].message.content
        code = extract_code(response_content)
        local_vars = {}
        exec(code, {"go": go}, local_vars)
        fig = local_vars.get("fig")
        return fig

    except Exception as e:
        return None


if __name__ == "__main__":
    question = "how many members in communit?"
    print(f"\n🔍 User question: {question}")

    # write SQL query
    sql_query_result = write_sql_query(question)
    print(f"💡 Generated SQL Query:\n{sql_query_result.sql_query}")

    # run SQL query
    sql_query_result = run_sql_query(sql_query_result.sql_query)
    print(f"💡 SQL Query Results:\n{sql_query_result}")

    # Generate final answer
    final_answer = write_answer(question, sql_query_result.to_string())
    print(f"💡 Final Answer:\n{final_answer}")
