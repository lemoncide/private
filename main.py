import argparse
import sys
import os
import asyncio

# Ensure the current directory is in the path
sys.path.append(os.getcwd())

from agent.core.graph import build_graph
from agent.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Agent CLI (LangGraph)")
    parser.add_argument("--task", type=str, help="The task for the agent to perform")
    parser.add_argument("--task-file", type=str, help="Read the task from a text file")
    
    args = parser.parse_args()
    
    task = args.task or "Calculate 10 + 5 and summarize."
    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            task = f.read().strip() or task
    
    print(f"Starting Agent with LangGraph Workflow for task: {task}")
    
    # Build the graph
    app = build_graph()
    
    # Initialize State
    initial_state = {
        "input": task,
        "plan": None,
        "current_step_index": 0,
        "past_steps": [],
        "context_variables": {},
        "response": None,
        "status": "pending",
        "error": None,
        "scratchpad": []
    }
    
    # Run the graph
    # stream() yields events, invoke() runs to completion
    try:
        final_state = asyncio.run(app.ainvoke(initial_state))
        
        print("\n--- Final Result ---\n")
        print(final_state.get("response", "No response generated."))
        
        # Optional: Print history
        # print("\n--- Execution History ---")
        # for step in final_state.get("past_steps", []):
        #     print(f"Step: {step['step']} -> Result: {step['result']}")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
