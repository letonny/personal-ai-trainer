import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import the CORS middleware
from pydantic import BaseModel as FastApiBaseModel
from typing import List, Dict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from .tools import search_tool, save_workout_plan  # Corrected relative import
from pathlib import Path

# Load the API key from the .env file
project_dir = Path(__file__).resolve().parent.parent
dotenv_path = project_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI(
    title="Pulse AI Personal Trainer",
    description="API for interacting with the LangChain AI agent.",
    version="1.0.0"
)

# --- Add CORS Middleware ---
# This allows your frontend (running on any origin "*") to communicate with this backend.
origins = ["*"]  # For development, allow all origins. For production, you'd list specific domains.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 1. Define the Structured Output (Pydantic Models) ---
class Exercise(FastApiBaseModel):
    name: str
    sets: int
    reps: str
    rest_period: str

class WorkoutPlan(FastApiBaseModel):
    title: str
    goal: str
    workout_type: str
    exercises: List[Exercise]
    final_note: str

class ChatRequest(FastApiBaseModel):
    query: str
    chat_history: List[Dict[str, str]]

# --- 2. Setup the "Brain" of the Agent ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
parser = PydanticOutputParser(pydantic_object=WorkoutPlan)
tools = [search_tool, save_workout_plan]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
            You are 'Pulse,' a world-class AI personal trainer. Your purpose is to help users achieve their fitness goals by creating effective, safe, and personalized workout plans. Your guiding principles are:
            1. Be Encouraging & Knowledgeable: Your tone should be motivating, supportive, and backed by fitness principles.
            2. Ask for Clarity: If a user's request is vague (e.g., "give me a workout"), you MUST ask clarifying questions to understand their specific goals (e.g., muscle gain, fat loss), experience level, available equipment, and time commitment. Do not proceed until you have this information.
            3. Use Your Tools: When you need more information about an exercise or want to save a plan, use the tools provided.
            4. Strictly Format Your Output: When you have all the necessary information and are ready to provide the workout plan, you MUST format your response according to the provided JSON schema. Do not add any other conversational text or pleasantries outside of this final, structured output.
            \n{format_instructions}
            """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Receives a chat message and returns the AI's response."""
    history_messages = []
    for msg in request.chat_history:
        if msg['type'] == 'human':
            history_messages.append(HumanMessage(content=msg['content']))
        elif msg['type'] == 'ai':
            history_messages.append(AIMessage(content=msg['content']))

    response = agent_executor.invoke({
        "query": request.query,
        "chat_history": history_messages
    })

    try:
        workout = parser.parse(response["output"])
        return {"type": "workout_plan", "data": workout.model_dump()}
    except Exception:
        return {"type": "message", "data": response["output"]}

# --- To run the server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)