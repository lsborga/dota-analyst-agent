import os
import gradio as gr
from dotenv import load_dotenv
import stratz
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# --- Environment Setup ---
# Load API keys from environment variables.
# For local testing, this command loads keys from the.env file.
# In a deployed environment (like Render), it loads from configured secrets.
load_dotenv()

# --- API Client Initialization ---
# Initialize the Stratz API client using the key from the environment.
# A try-except block handles the case where the key might not be set,
# preventing the app from crashing on startup if secrets are missing.
try:
    stratz_api = stratz.Api(token=os.getenv("STRATZ_API_KEY"))
except Exception:
    stratz_api = None

# --- Agent Tool Definition ---
# This function serves as the "action" for our AI agent. It fetches and
# processes raw data from the Stratz API.
def get_match_analysis(query: str) -> str:
    """
    Analyzes a Dota 2 match for a specific player.
    The input for this function MUST be a single string containing the
    match_id and the player's steam_id, separated by a comma.
    Example: "6279293344, 91064780"
    """
    if not stratz_api:
        return "Stratz API client is not initialized. Please check API key."
        
    try:
        # The agent will pass a single string; we parse it here.
        #.strip() is used to remove any accidental whitespace.
        match_id_str, steam_id_str = query.split(',')
        match_id = int(match_id_str.strip())
        player_steam_id = int(steam_id_str.strip())
    except ValueError:
        # This error occurs if the input string is not in the expected format.
        return "Invalid input format. Please provide match_id and steam_id separated by a comma."

    try:
        # Fetch the full match data from the Stratz API.
        match = stratz_api.get_match(match_id)
        
        # Find the specific player's data within the match.
        player_data = None
        for p in match.players:
            if p.steam_account_id == player_steam_id:
                player_data = p
                break
        
        if not player_data:
            return f"Player with Steam ID {player_steam_id} not found in match {match_id}."

        # Structure the raw data into a clean dictionary for the agent.
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
        
        # Return the structured data as a string for the LLM to process.
        return f"Analysis for player {player_data.steam_account.name} in match {match_id}: {analysis}"

    except Exception as e:
        return f"An error occurred while fetching match data: {e}"

# --- Main Agent Function ---
# This function is the entry point called by the Gradio interface.
# It orchestrates the entire process of initializing and running the agent.
def get_player_analysis(match_id, steam_id):
    """
    Initializes and runs the AI agent to analyze a Dota 2 match.
    """
    # Pre-flight check for API keys to provide a clear error message.
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("STRATZ_API_KEY"):
        return "Error: API keys are not configured. Please ensure they are set in the environment secrets."

    # Initialize the LLM (the "brain" of the agent).
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Define the tools the agent can use. This is the crucial link between
    # the agent's reasoning and its ability to act.
    tools = [
    Tool(
        name="Dota2MatchAnalyzer",
        func=get_match_analysis,
        description="""
        Analyzes a Dota 2 match for a specific player.
        The input for this function MUST be a single string containing the
        match_id and the player's steam_id, separated by a comma.
        Example: "6279293344, 91064780"
        """
    )
]

    # Define the agent's persona and high-level instructions.
    system_prompt = """
    You are an expert Dota 2 analyst. Your goal is to provide detailed, actionable advice to help a player improve based on their match data.
    When analyzing a match, focus on key metrics like KDA, GPM, XPM, last hits, and item choices.
    Relate these stats back to the hero's typical strategies and timings.
    Provide clear, concise, and constructive feedback.
    """

    # Initialize the agent, giving it the LLM, the tools, and the system prompt.
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"prefix": system_prompt}
    )

    # Construct the specific goal for this run, using the user's input.
    goal = f"""
    Please provide a strategic analysis of the player's performance in match {match_id} for the player with Steam ID {steam_id}.
    Use the Dota2MatchAnalyzer tool to get the data.
    Based on the data, explain their performance, highlighting strengths and areas for improvement.
    """

    try:
        # Execute the agent with the defined goal.
        response = agent.run(goal)
        return response
    except Exception as e:
        return f"An error occurred while running the agent: {e}"

# --- Gradio Web Interface ---
# This section defines the user-facing web application.
iface = gr.Interface(
    fn=get_player_analysis,
    # The 'inputs' list must match the parameters of the 'fn' function.
    # Each component in this list corresponds to one function parameter, in order.
    inputs=[
    gr.Textbox(label="Match ID"),
    gr.Textbox(label="Player Steam32 ID")
]
    # The 'outputs' component displays the final result from the agent.
    outputs=gr.Textbox(label="AI Analyst Report", lines=20),
    title="Dota 2 AI Analyst Agent",
    description="Enter a Match ID and your Steam32 ID to get a detailed performance analysis from an AI expert. You can find your Steam32 ID on your Stratz.com profile.",
    allow_flagging="never"
)

# --- Launch the App ---
# This block allows the script to be run directly for local testing (e.g., python app.py).
# When deployed on a service like Render, the Gunicorn server will
# directly access the 'iface' object, so this block is not executed.
if __name__ == "__main__":
    iface.launch()