from langchain_core.tools import tool
import json

# The AI agent can choose to use any of these functions when it needs them.
# The docstring for each function is very important, as it tells the AI what the tool does.

@tool
def search_tool(query: str) -> str:
    """
    Use this tool to search for up-to-date information on exercises, fitness concepts,
    or nutrition. It's useful for when you need to verify exercise form or find new workout ideas.
    """
    print(f"--> [Tool Called] Searching for: '{query}'...")
    # In a real application, this would call a search API (like Google Search).
    # For this example, we'll return a placeholder string.
    return f"Search results for '{query}': (Placeholder) The bench press is a compound exercise that targets the pectoralis major..."

@tool
def save_workout_plan(plan: str, filename: str) -> str:
    """
    Use this tool to save the generated workout plan as a JSON file.
    The 'plan' should be a valid JSON string.
    The 'filename' should be a descriptive name ending in .json.
    """
    print(f"--> [Tool Called] Saving workout plan to '{filename}'...")
    try:
        # The AI will pass the plan as a string, so we parse it back into a dict
        workout_data = json.loads(plan)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(workout_data, f, indent=4)
        return f"Successfully saved the workout plan to '{filename}'."
    except Exception as e:
        return f"Error: Could not save the file. Reason: {e}"
