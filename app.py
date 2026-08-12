
import sys
import io
import traceback
import os

from typing import TypedDict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
# 7. TEST CASE GENERATOR
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate 3 to 5 specific test scenarios for a coding task.
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
    Developer agent that generates Python code.
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
                parts.append(item.get("text", ""))
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
    Manager node that prepares the final report.
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
# 12. LANGGRAPH WORKFLOW
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
    Route manager result to archiver.
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


agent = workflow.compile()


# ============================================================
# 13. WEB UI
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>AI Coding Agent</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
}

.container {
    width: 90%;
    max-width: 1100px;
    margin: 40px auto;
}

.header {
    text-align: center;
    margin-bottom: 35px;
}

.header h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.header p {
    color: #94a3b8;
    font-size: 17px;
}

.card {
    background: #1e293b;
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 25px;
    border: 1px solid #334155;
}

label {
    display: block;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 12px;
}

textarea {
    width: 100%;
    min-height: 150px;
    resize: vertical;
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #475569;
    border-radius: 10px;
    padding: 15px;
    font-size: 16px;
    outline: none;
}

textarea:focus {
    border-color: #38bdf8;
}

button {
    margin-top: 15px;
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: 10px;
    background: #e11d48;
    color: white;
    font-size: 17px;
    font-weight: bold;
    cursor: pointer;
}

button:hover {
    background: #be123c;
}

button:disabled {
    background: #475569;
    cursor: not-allowed;
}

.section-title {
    font-size: 20px;
    margin-bottom: 12px;
}

pre {
    background: #020617;
    border-radius: 10px;
    padding: 18px;
    overflow-x: auto;
    white-space: pre-wrap;
    color: #cbd5e1;
    line-height: 1.5;
}

.status {
    text-align: center;
    margin-top: 15px;
    color: #38bdf8;
}

.hidden {
    display: none;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 30px;
}

</style>

</head>


<body>

<div class="container">

    <div class="header">

        <h1>🤖 AI Coding Agent</h1>

        <p>
            Multi-Agent Python Developer & Tester
        </p>

    </div>


    <div class="card">

        <label for="task">
            Enter Your Coding Task
        </label>

        <textarea
            id="task"
            placeholder="Example: Create a Python program to check whether a number is prime."
        ></textarea>

        <button
            id="runButton"
            onclick="runAgent()"
        >
            Run AI Agent
        </button>

        <div id="status" class="status"></div>

    </div>


    <div id="results" class="hidden">


        <div class="card">

            <div class="section-title">
                👨‍💻 Generated Python Code
            </div>

            <pre id="code"></pre>

        </div>


        <div class="card">

            <div class="section-title">
                🧪 Test Scenarios
            </div>

            <pre id="tests"></pre>

        </div>


        <div class="card">

            <div class="section-title">
                📊 Execution Report
            </div>

            <pre id="report"></pre>

        </div>


    </div>


    <div class="footer">

        Built with FastAPI + LangGraph + Gemini

    </div>

</div>


<script>

async function runAgent() {

    const task =
        document.getElementById("task").value.trim();

    const button =
        document.getElementById("runButton");

    const status =
        document.getElementById("status");

    const results =
        document.getElementById("results");


    if (!task) {

        status.textContent =
            "Please enter a coding task.";

        return;
    }


    button.disabled = true;

    button.textContent = "Running AI Agent...";

    status.textContent =
        "Developer and Tester agents are working...";

    results.classList.add("hidden");


    try {

        const response = await fetch("/run", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                task: task
            })

        });


        const data = await response.json();


        if (!data.success) {

            throw new Error(
                data.error || "Agent failed."
            );

        }


        document.getElementById("code").textContent =
            data.code || "No code generated.";


        document.getElementById("tests").textContent =
            data.test_cases || "No test cases generated.";


        document.getElementById("report").textContent =
            data.report || "No report generated.";


        results.classList.remove("hidden");

        status.textContent =
            "✓ AI Agent completed successfully.";


    } catch (error) {

        status.textContent =
            "Error: " + error.message;

    }


    button.disabled = false;

    button.textContent = "Run AI Agent";

}

</script>


</body>

</html>
"""


# ============================================================
# 14. UI ROUTE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return HTML_PAGE


# ============================================================
# 15. API ENDPOINT
# ============================================================

@app.post("/run")
def run_agent(request: CodingTask):
    """
    Run the AI coding agent.
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
