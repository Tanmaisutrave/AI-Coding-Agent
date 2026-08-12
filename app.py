
import sys
import io
import traceback
import os

from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. API KEY
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)


# ============================================================
# 3. FASTAPI
# ============================================================

app = FastAPI(
    title="AI Coding Agent",
    description="Multi-agent AI coding and testing system",
    version="1.0.0"
)


# ============================================================
# 4. REQUEST MODEL
# ============================================================

class CodingTask(BaseModel):
    task: str


# ============================================================
# 5. STATE
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    test_cases: Optional[str]
    report: Optional[str]


# ============================================================
# 6. PYTHON EXECUTION TOOL
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code and return the output or error.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        exec(clean_code, {})

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    if result.strip():

        return result.strip()

    return "Program executed successfully with no output."


# ============================================================
# 7. TEST CASE GENERATOR TOOL
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate 3 to 5 specific test scenarios
    for the given coding task.
    """

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 specific test scenarios
for the following Python coding task:

{task_description}

Include:

1. Normal cases
2. Edge cases
3. Invalid input cases if applicable

Return only a numbered list.
"""

    response = llm.invoke(prompt)

    return str(response.content)


# ============================================================
# 8. DEVELOPER AGENT
# ============================================================

def developer_node(state: CrewState):
    """
    Developer agent that generates Python code
    for the requested coding task.
    """

    task = state["messages"][-1].content

    developer_prompt = f"""
You are an expert Python Developer.

Solve the following programming task:

{task}

Requirements:

- Write clean Python code.
- Make the program executable.
- Handle common edge cases.
- Use simple and understandable Python.
- Do not explain the code.
- Return ONLY the Python code.
"""

    response = llm.invoke(developer_prompt)

    content = response.content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                parts.append(
                    item.get("text", "")
                )

            else:

                parts.append(str(item))

        code = "\n".join(parts)

    else:

        code = str(content)

    code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    return {
        "code": code
    }


# ============================================================
# 9. TESTER AGENT
# ============================================================

def tester_node(state: CrewState):
    """
    Tester agent that generates test scenarios
    and executes the generated Python code.
    """

    task = state["messages"][-1].content

    code = state["code"]

    test_cases = generate_test_cases.invoke(task)

    execution_result = run_python_code.invoke(
        {
            "code": code
        }
    )

    report = f"""
EXECUTION RESULT
----------------

{execution_result}


TEST SCENARIOS
--------------

{test_cases}
"""

    return {
        "test_cases": test_cases,
        "report": report
    }


# ============================================================
# 10. MANAGER
# ============================================================

def manager_node(state: CrewState):
    """
    Manager node that prepares the final testing report.
    """

    report = state.get(
        "report",
        "No report available."
    )

    return {
        "report": report,
        "next_step": "complete"
    }


# ============================================================
# 11. ARCHIVER
# ============================================================

def archiver_node(state: CrewState):
    """
    Archive the completed coding task.
    """

    return {
        "next_step": "exit"
    }


# ============================================================
# 12. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(CrewState)


workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_node(
    "manager",
    manager_node
)

workflow.add_node(
    "archiver",
    archiver_node
)


workflow.add_edge(
    START,
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    "manager"
)


def route_after_manager(state: CrewState):
    """
    Route the workflow from manager to archiver.
    """

    return "archiver"


workflow.add_conditional_edges(
    "manager",
    route_after_manager
)

workflow.add_edge(
    "archiver",
    END
)


# ============================================================
# 13. COMPILE
# ============================================================

agent = workflow.compile()


# ============================================================
# 14. API ENDPOINTS
# ============================================================

@app.get("/")
def home():
    """
    Health check endpoint.
    """

    return {
        "status": "online",
        "message": "AI Coding Agent is running."
    }


@app.post("/run")
def run_agent(request: CodingTask):
    """
    Run the AI coding agent for a given task.
    """

    task = request.task.strip()

    if not task:

        return {
            "success": False,
            "error": "Coding task cannot be empty."
        }

    initial_state = {

        "messages": [
            HumanMessage(content=task)
        ],

        "next_step": None,

        "code": None,

        "test_cases": None,

        "report": None
    }

    try:

        result = agent.invoke(
            initial_state,
            config={
                "recursion_limit": 20
            }
        )

        return {

            "success": True,

            "task": task,

            "code": result.get("code"),

            "test_cases": result.get("test_cases"),

            "report": result.get("report")
        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)
        }
