import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with GitHub Models
client = OpenAI(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
)


# Load context from external text file
file_path = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "community_generic_info.txt"
)
with open(file_path, "r", encoding="utf-8") as file:
    context_text = file.read()

# Developer prompt defining the assistant's role and behavior
DEVELOPER_PROMPT = """
# Identity
You are a helpful assistant that provides information only about the Dabburiya Tech community based on the provided context.

# Instructions
The assistant answers only questions related to the purpose of this chat. If the user asks something outside the chat purpose, reply:"The question is outside the purpose of this chat."
If the question is related to the chat purpose but the answer is not found in the provided CONTEXT, reply: "Not in provided context."
The assistant never invents facts not present in the CONTEXT.

# Examples
"""

# User prompt template for dynamic question insertion
USER_PROMPT_TEMPLATE = """
You are being asked a question about the Dabburiya Tech community.

QUESTION:
{question}

CONTEXT:
{context}

INSTRUCTIONS:
Answer the QUESTION using only the CONTEXT.
If the answer is not found in the CONTEXT, reply: "Not in provided context".
"""


# Function to run chat completion
def run_chat(question: str) -> str:

    USER_PROMPT = USER_PROMPT_TEMPLATE.format(question=question, context=context_text)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DEVELOPER_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    question = "what the community trying to empower?"
    print(f"\n🔍 User question: {question}")

    # Get community advice
    response = run_chat(question)
    print(f"💡 Chat Response:\n{response}")
