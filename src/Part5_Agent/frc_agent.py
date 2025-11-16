import os
import asyncio
from dotenv import load_dotenv
import requests
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# Load environment variables from .env file
load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TBA_KEY = os.getenv("TBA_KEY")

# os.environ.get("TBA_KEY")  # get your key from your TBA account settings
headers = {"X-TBA-Auth-Key": TBA_KEY}


@function_tool
def get_awards_by_team(team_id: int):
    # Get ALL awards for the team by their team number, return event details like event_key of the awards
    url = f"https://www.thebluealliance.com/api/v3/team/frc{team_id}/awards"
    awards = requests.get(url, headers=headers).json()
    return awards


@function_tool
def get_matches_by_team_and_event(team_id: int, event_key: str):
    # Get ALL matches for the team at a specific event including match key, without the points breakdown
    url = f"https://www.thebluealliance.com/api/v3/team/frc{team_id}/event/{event_key}/matches"
    matches = requests.get(url, headers=headers).json()
    for match in matches:
        match.pop("score_breakdown", None)
    return matches


@function_tool
def get_match_by_key(match_key: str):
    # Get match details by match key all the points breakdown
    url = f"https://www.thebluealliance.com/api/v3/match/{match_key}"
    match = requests.get(url, headers=headers).json()
    return match


client = AsyncOpenAI(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
)
set_default_openai_client(client)  # , use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

assistant = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    instructions="You are a helpful assistant for FRC teams.",
    tools=[get_awards_by_team, get_matches_by_team_and_event, get_match_by_key],
)


def run_frc_agent(question: str):
    result = Runner.run_sync(
        assistant,
        question,
    )
    return result


if __name__ == "__main__":
    question = "How many match points in auto mode of the last match when team 5715 won their last championship award?"
    print(f"\n🔍 User question: {question}")

    result = run_frc_agent(question)
    print(f"\n💡 Agent response: {result.final_output}")
