import os
import gradio as gr
from dotenv import load_dotenv
import stratz
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI

# Load API keys from environment variables
# For local testing, it loads from the.env file. For deployment, from secrets.
load_dotenv()

# Initialize the Stratz API client
# It will automatically use the STRATZ_API_KEY environment variable
try:
    stratz_api = stratz.Api()
except Exception:
    # Handle case where API key might not be set initially
    stratz_api = None

# --- Agent Tool Definition ---
def get_match_analysis(query: str) -> str:
    """
    Analyzes a Dota 2 match for a specific player.
    The input should be a string containing the match_id and the player's steam_id,
    separated by a comma. Example: "6279293344, 91064780"
    """
    if not stratz_api:
        return "Stratz API client is not initialized. Please check API key."
        
    try:
        match_id_str, steam_id_str = query.split(',')
        match_id = int(match_id_str.strip())
        player_steam_id = int(steam_id_str.strip())
    except ValueError:
        return "Invalid input format. Please provide match_id and steam_id separated by a comma."

    try:
        match = stratz_api.get_match(match_id)
        
        player_data = None
        for p in match.players:
            if p.steam_account_id == player_steam_id:
                player_data = p
                break
        
        if not player_data:
            return f"Player with Steam ID {player_steam_id} not found in match {match_id}."

        analysis = {
            "hero": player_data.hero.display_name,
            "kills": player_data.num_kills,
            "deaths": player_data.num_deaths,
            "assists": player_data.num_assists,
            "kda": round((player_data.num_kills + player_data.num_assists) / (player_data.num_deaths or 1), 2),
            "gold_per_min": player_data.gold_per_minute,
            "xp_per_min": player_data.xp_per_minute,
            "last_hits": player_data.num_last_hits,
            "denies": player_data.num_denies,
            "lane_performance": player_data.lane_outcome.name if player_data.lane_outcome else "N/A",
            "final_items": [item.item.display_name for item in player_data.final_items]
        }
        
        return f"Analysis for player {player_data.steam_account.name} in match {match_id}: {analysis}"

    except Exception as e:
        return f"An error occurred while fetching match data: {e}"

# --- Main Agent Function ---
def get_player_analysis(match_id, steam_id):
    """
    This function initializes and runs the AI agent to analyze a Dota 2 match.
    """
    # Check if API keys are available
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("STRATZ_API_KEY"):
        return "Error: API keys are not configured. Please ensure they are set in the environment secrets."

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Define the tools the agent can use
    tools =

    # Define the agent's persona and instructions
    system_prompt = """
    You are an expert Dota 2 analyst. Your goal is to provide detailed, actionable advice to help a player improve based on their match data.
    When analyzing a match, focus on key metrics like KDA, GPM, XPM, last hits, and item choices.
    Relate these stats back to the hero's typical strategies and timings.
    Provide clear, concise, and constructive feedback.
    """

    # Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"prefix": system_prompt}
    )

    # Construct the goal for the agent
    goal = f"""
    Please provide a strategic analysis of the player's performance in match {match_id} for the player with Steam ID {steam_id}.
    Use the Dota2MatchAnalyzer tool to get the data.
    Based on the data, explain their performance, highlighting strengths and areas for improvement.
    """

    try:
        response = agent.run(goal)
        return response
    except Exception as e:
        return f"An error occurred while running the agent: {e}"

# --- Gradio Web Interface ---
iface = gr.Interface(
    fn=get_player_analysis,
    inputs=,
    outputs=gr.Textbox(label="AI Analyst Report", lines=20),
    title="Dota 2 AI Analyst Agent",
    description="Enter a Match ID and your Steam32 ID to get a detailed performance analysis from an AI expert. You can find your Steam32 ID on your Stratz.com profile.",
    allow_flagging="never"
)

# --- Launch the App ---
# When deploying, gunicorn will call 'iface' directly.
# This block is for local testing.
if __name__ == "__main__":
    iface.launch()