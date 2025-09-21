import os
import gradio as gr
from dotenv import load_dotenv
import stratz
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Environment Setup ---
# Load API keys from environment variables.
# For local testing, this command loads keys from the.env file.
# In a deployed environment (like Render), it loads from configured secrets.
load_dotenv()


# --- Agent Tool Definition ---
# This function serves as the "action" for our AI agent. It fetches and
# processes raw data from the Stratz API.
# --- Agent Tool Definition ---
def get_match_analysis(query: str) -> str:
    """
    Analyzes a Dota 2 match for a specific player.
    The input for this function MUST be a single string containing the
    match_id and the player's steam_id, separated by a comma.
    Example: "8471429194,137863862"
    """
    try:
        # The agent will pass a single string; we parse it here.
        match_id_str, steam_id_str = query.split(',')
        match_id = int(match_id_str.strip())
        player_steam_id = int(steam_id_str.strip())
    except ValueError:
        return "Invalid input format. Please provide match_id and steam_id separated by a comma."

    try:
        # --- NEW: Initialize the API client inside the function ---
        stratz_api = stratz.Api(token=os.getenv("STRATZ_API_KEY"))
        
        # Fetch the full match data from the Stratz API.
        match = stratz_api.get_match(match_id)
        
        # Find the specific player's data within the match.
        player_data = None
        for p in match.players:
            if p.steam_account_id == player_steam_id:
                player_data = p
        
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
# --- Main Agent Function ---
def get_player_analysis(match_id, steam_id):
    """
    Initializes and runs the AI agent to analyze a Dota 2 match.
    """
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("STRATZ_API_KEY"):
        return "Error: API keys are not configured. Please ensure GOOGLE_API_KEY and STRATZ_API_KEY are set."

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.getenv("GOOGLE_API_KEY"))

    tools = [
        Tool(
            name="Dota2MatchAnalyzer",
            func=get_match_analysis,
            description="""
            Analyzes a Dota 2 match for a specific player.
            The input for this function MUST be a single string containing the
            match_id and the player's steam_id, separated by a comma.
            Example: "8471429194,137863862"
            """
        )
    ]

    # This is the new, more reliable prompt structure
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert Dota 2 analyst. Your goal is to provide detailed, actionable advice to help a player improve based on their match data.
        When analyzing a match, focus on key metrics like KDA, GPM, XPM, last hits, and item choices.
        Relate these stats back to the hero's typical strategies and timings.
        Provide clear, concise, and constructive feedback. You MUST get your data from the Dota2MatchAnalyzer tool.
        """),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # This is the new, more modern way to build the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # The user's goal for the agent
    goal = f"""
    Please provide a strategic analysis of the player's performance in match {match_id} for the player with Steam ID {steam_id}.
    Do not make up any information. Use the Dota2MatchAnalyzer tool to get all the data.
    Based on the real data from the tool, explain their performance, highlighting strengths and areas for improvement.
    """

    try:
        # Execute the agent and get the final response
        response = agent_executor.invoke({"input": goal})
        return response['output']
    except Exception as e:
        return f"An error occurred while running the agent: {e}"

# --- Gradio Web Interface ---
# This section defines the user-facing web application.
iface = gr.Interface(
    fn=get_player_analysis,
    inputs=[
        gr.Textbox(label="Match ID"),
        gr.Textbox(label="Player Steam32 ID")
    ],
    outputs=gr.Textbox(label="AI Analyst Report", lines=20),
    title="Dota 2 AI Analyst Agent",
    description="Enter a Match ID and your Steam32 ID to get a detailed performance analysis from an AI expert. You can find your Steam32 ID on your Stratz.com profile.",
    allow_flagging="never"
)

# --- FastAPI App Mounting for Deployment ---
# We wrap the Gradio app in a FastAPI app to add a custom health check endpoint.
from fastapi import FastAPI, Response

app = FastAPI()

# This is the dedicated endpoint for the health checker
@app.get("/health")
def health_check():
    """Returns a 200 OK response for health checks."""
    return Response(status_code=200, content="OK")

# Mount the Gradio interface onto the FastAPI app at the root URL
app = gr.mount_gradio_app(app, iface, path="/")

# The __main__ block is for local testing (optional, not used by Render)
if __name__ == "__main__":
    import uvicorn
    # Note: Running this locally requires installing fastapi and uvicorn
    # pip install fastapi "uvicorn[standard]"
    uvicorn.run(app, host="0.0.0.0", port=7860)