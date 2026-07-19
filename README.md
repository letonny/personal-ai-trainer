# 💪 Pulse AI Fitness Trainer

---

An AI-powered personal fitness trainer built with **Python**, **LangChain**, **FastAPI**, and **OpenAI GPT-4o** that generates personalized workout plans through natural language conversations. The application demonstrates LLM orchestration, structured AI output using Pydantic, tool calling, and external API integration to provide customized fitness recommendations.

***

## Features

---

- Personalized AI-generated workout plans
- Natural language fitness conversations
- LangChain tool-calling agent
- Structured output validation using Pydantic
- Real-time exercise research using DuckDuckGo Search
- Fitness concept lookup with Wikipedia
- Save generated workout plans to a text file
- Secure API key management using environment variables

***

## Technologies

---

- Python
- FastAPI
- LangChain
- OpenAI GPT-4o API
- Pydantic
- DuckDuckGo Search API
- Wikipedia API
- python-dotenv

***

## Architecture

---

```text
                    User Fitness Request
                             │
                             ▼
                 +-----------------------+
                 |      FastAPI / CLI    |
                 +-----------------------+
                             │
                             ▼
                +-------------------------+
                |  LangChain Agent        |
                | (GPT-4o + Tool Calling) |
                +-------------------------+
                   │        │        │
                   │        │        │
                   ▼        ▼        ▼
           DuckDuckGo   Wikipedia   Save Tool
             Search       Lookup     (.txt)
                   │        │
                   └────┬───┘
                        ▼
              +--------------------+
              |  GPT-4o Reasoning  |
              +--------------------+
                        │
                        ▼
         Pydantic Output Validation
                        │
                        ▼
        Personalized Workout Plan
```

***

## AI Workflow

---

1. User enters a natural language fitness request.
2. LangChain sends the request to GPT-4o.
3. The agent determines whether external tools are needed.
4. If necessary, it searches DuckDuckGo or Wikipedia for additional information.
5. GPT-4o generates a structured workout plan.
6. Pydantic validates the response against predefined schemas.
7. The workout plan is displayed to the user and can optionally be saved to a text file.

***

## Example Prompt

---

```text
Create a 4-day beginner muscle-building workout plan using only dumbbells.
```

### Example Output

```text
Goal: Muscle Gain

Workout Split:
Upper / Lower

Exercises
- Dumbbell Bench Press
- Goblet Squat
- Romanian Deadlift
- Shoulder Press
- Bent-Over Row

Motivational Tip
Consistency beats perfection. Focus on improving each week.
```

***

## AI Components

---

| Component | Purpose |
|-----------|---------|
| GPT-4o | Generates personalized workout plans |
| LangChain Agent | Orchestrates reasoning and tool usage |
| DuckDuckGo Search | Retrieves current fitness information |
| Wikipedia | Provides fitness concept explanations |
| Pydantic | Validates structured AI output |
| Save Tool | Stores workout plans locally |

***

## Future Improvements

---

- Web dashboard built with Next.js
- User authentication
- Workout history database
- Nutrition and meal planning
- Exercise image and video recommendations
- Progress tracking and analytics
- RAG using a custom exercise knowledge base
- Streaming AI responses
- Multi-agent workflow for coaching and nutrition
