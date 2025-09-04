import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
# Assuming your tools are in a file named tools.py
from tools import search_tool, wiki_tool, save_tool

# Load environment variables from a .env file
load_dotenv()

# --- 1. Redefined Pydantic Models for a Workout Plan ---
# This structure defines what a single exercise looks like.
class Exercise(BaseModel):
    name: str = Field(description="The specific name of the exercise, e.g., 'Barbell Squat'.")
    sets: int = Field(description="The number of sets to perform.")
    reps: str = Field(description="The target repetition range, e.g., '8-12'.")
    rest_period: str = Field(description="The recommended rest time between sets, e.g., '60 seconds'.")

# This is the main output structure for the entire workout plan.
class WorkoutPlan(BaseModel):
    goal: str = Field(description="The primary fitness goal this plan is designed for, e.g., 'Muscle Gain'.")
    plan_name: str = Field(description="A catchy or descriptive name for the workout plan, e.g., 'Foundational Strength'.")
    workout_split: str = Field(description="The type of workout split, e.g., 'Full Body', 'Push/Pull/Legs'.")
    exercises: List[Exercise] = Field(description="A list of exercises included in the workout plan.")
    motivational_tip: str = Field(description="A brief, encouraging tip to motivate the user.")

# --- 2. Setup the Language Model and Parser (Now with OpenAI) ---
# Using OpenAI's gpt-4o model.
llm = ChatOpenAI(model="gpt-4o")

# The parser will ensure the LLM's output conforms to our WorkoutPlan model.
parser = PydanticOutputParser(pydantic_object=WorkoutPlan)

# --- 3. Updated System Prompt for a Personal Trainer Persona ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are 'Aether,' a world-class AI personal trainer. Your goal is to help users achieve their fitness goals
            by creating effective, safe, and personalized workout plans.

            - Your tone is encouraging, knowledgeable, and motivating. 🦾
            - If the user's query is vague (e.g., "give me a workout"), you MUST ask clarifying questions to understand
              their goals (e.g., muscle gain, fat loss), experience level, available equipment, and time commitment.
            - Use the provided tools to research exercises or fitness concepts if needed.
            - You MUST structure your final response using the required format. Provide no other text or conversation
              outside of the specified JSON format.
            \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# --- 4. Agent and Executor Setup (Largely Unchanged) ---
# The tools are still useful for looking up exercise info or saving the plan.
tools = [search_tool, wiki_tool, save_tool]

# Create the agent that combines the LLM, prompt, and tools.
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# The executor runs the agent and handles the logic.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5. User Interaction ---
# Updated input prompt to match the new persona.
query = input("What is your fitness goal today? (e.g., 'Create a 3-day beginner muscle-building plan for the gym') ")

# Invoke the agent with the user's query.
raw_response = agent_executor.invoke({"query": query})

# --- 6. Parse and Print the Output ---
try:
    # The output from the agent needs to be parsed into our Pydantic model.
    # Note: The structure might be nested, so we access ["output"].
    structured_response = parser.parse(raw_response.get("output"))

    # Print the structured data in a readable way
    print("\n--- Your Workout Plan ---")
    print(f"🔥 Goal: {structured_response.goal}")
    print(f"💪 Plan: {structured_response.plan_name} ({structured_response.workout_split})")
    print("\n## Exercises ##")
    for ex in structured_response.exercises:
        print(f"- {ex.name}: {ex.sets} sets of {ex.reps} reps, with {ex.rest_period} rest.")
    print("\n✨ Motivational Tip ✨")
    print(structured_response.motivational_tip)

except Exception as e:
    print("\n--- Error ---")
    print("Could not parse the AI's response. This can happen if the query is too complex or vague.")
    print(f"Error details: {e}")
    print("\nRaw AI Response:")
    print(raw_response.get("output"))