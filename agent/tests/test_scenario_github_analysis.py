import asyncio
import os
import sys
import pytest
import shutil as _shutil

sys.path.append(os.getcwd())

from agent.core.agent import LangGraphAgent
from agent.utils.config import config
from agent.utils.logger import logger

def run_github_analysis():
    print("Initializing Agent for GitHub Analysis Scenario...")
    
    agent = LangGraphAgent()
    
    objective = """
    1. Search for a GitHub repository named 'modelcontextprotocol/servers'.
    2. List the files in the repository to find a Python file.
    3. Read the content of 'README.md' from that repository.
    4. Summarize the README content.
    5. Save the summary to 'github_summary.txt' in the sandbox.
    """
    
    print(f"\nRunning Agent with objective: {objective}")

    target_file = os.path.join(os.getcwd(), "sandbox", "github_summary.txt")
    if os.path.exists(target_file):
        os.remove(target_file)
    
    try:
        final_state = asyncio.run(agent.ainvoke(objective))
        
        print("\nExecution Completed.")
        
        trace = final_state.get("trace", {})
        execution_history = trace.get("past_steps", [])
        
        print(f"Execution History Length: {len(execution_history)}")
        
        search_called = False
        read_called = False
        write_called = False
        
        for step_result in execution_history:
            if hasattr(step_result, "model_dump"):
                step_dict = step_result.model_dump()
            else:
                step_dict = step_result

            print(f"Step {step_dict.get('step_id')}: {step_dict.get('status')}")
            meta = step_dict.get('meta', {})
            if meta:
                tool_name = meta.get("tool", "")
                print(f"  Tool: {tool_name}")
                if "search_repositories" in tool_name: search_called = True
                if "get_file_contents" in tool_name: read_called = True
                if "write_file" in tool_name: write_called = True

            if step_dict.get("status") == "failed":
                print(f"  ErrorType: {step_dict.get('error_type')}")
                print(f"  Error: {step_dict.get('error')}")
        
        if search_called:
            print("SUCCESS: GitHub Search tool was called.")
        else:
            print("FAILURE: GitHub Search tool was NOT called.")
            
        if write_called and os.path.exists(target_file):
             print(f"SUCCESS: Summary file created at {target_file}")
        else:
             print("FAILURE: Summary file not created.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

def _assert_contract(final_state):
    trace = final_state.get("trace", {})
    execution_history = trace.get("past_steps", [])

    if not execution_history:
        raise AssertionError("No execution history produced")

    for step_result in execution_history:
        step_dict = step_result.model_dump() if hasattr(step_result, "model_dump") else step_result
        if step_dict.get("status") != "success":
            raise AssertionError(f"Step failed: {step_dict.get('step_id')} error={step_dict.get('error')}")
        if step_dict.get("error_type") == "schema_error":
            raise AssertionError(f"Schema error in step {step_dict.get('step_id')}: {step_dict.get('error')}")

    target_file = os.path.join(os.getcwd(), "sandbox", "github_summary.txt")
    if not os.path.exists(target_file):
        raise AssertionError(f"Expected summary file not created: {target_file}")


def test_github_analysis_offline_contract(offline_llm_and_mcp):
    agent = LangGraphAgent()

    objective = """
    1. Search for a GitHub repository named 'modelcontextprotocol/servers'.
    2. List the files in the repository to find a Python file.
    3. Read the content of 'README.md' from that repository.
    4. Summarize the README content.
    5. Save the summary to 'github_summary.txt' in the sandbox.
    """

    target_file = os.path.join(os.getcwd(), "sandbox", "github_summary.txt")
    if os.path.exists(target_file):
        os.remove(target_file)

    final_state = asyncio.run(agent.ainvoke(objective))
    _assert_contract(final_state)


def test_github_analysis():
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("integration test")

    if not os.environ.get("GITHUB_TOKEN"):
        pytest.skip("missing GITHUB_TOKEN")

    if not (_shutil.which("npx") or _shutil.which("npx.cmd")):
        pytest.skip("missing npx")

    agent = LangGraphAgent()
    objective = """
    1. Search for a GitHub repository named 'modelcontextprotocol/servers'.
    2. List the files in the repository to find a Python file.
    3. Read the content of 'README.md' from that repository.
    4. Summarize the README content.
    5. Save the summary to 'github_summary.txt' in the sandbox.
    """

    target_file = os.path.join(os.getcwd(), "sandbox", "github_summary.txt")
    if os.path.exists(target_file):
        os.remove(target_file)

    final_state = asyncio.run(agent.ainvoke(objective))
    _assert_contract(final_state)

if __name__ == "__main__":
    missing = []
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        missing.append("RUN_INTEGRATION_TESTS=1")
    if not os.environ.get("GITHUB_TOKEN"):
        missing.append("GITHUB_TOKEN")
    if not (_shutil.which("npx") or _shutil.which("npx.cmd")):
        missing.append("npx")

    if missing:
        print("Skipped: integration prerequisites not met.")
        print("Missing:", ", ".join(missing))
        raise SystemExit(0)

    run_github_analysis()
