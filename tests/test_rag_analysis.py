import sys
import os
import asyncio
import json
import logging

# Configure Logging to be less noisy
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Add project root to path
sys.path.append(os.getcwd())

from agent.core.planner import Planner
from agent.core.validator import PlanValidator
from agent.core.executor import ToolExecutor
from agent.core.schema import ExecutionContext
from agent.tools.manager import ToolManager
from agent.memory.manager import MemoryManager

# Mocking the existence of the RAG directory for the test if it doesn't exist
RAG_DIR = os.path.join(os.getcwd(), "tests", "mock_rag")
os.makedirs(RAG_DIR, exist_ok=True)
with open(os.path.join(RAG_DIR, "README.md"), "w") as f:
    f.write("# RAG System\nThis is a retrieval augmented generation system.")
with open(os.path.join(RAG_DIR, "main.py"), "w") as f:
    f.write("print('Starting RAG system...')")

async def run_integration_test():
    print("=== Starting RAG Analysis Integration Test ===")
    
    # 1. Initialize Components
    print("\n[1] Initializing Components...")
    memory = MemoryManager()
    tool_manager = ToolManager(memory_manager=memory)
    planner = Planner()
    validator = PlanValidator()
    executor = ToolExecutor(tool_manager)
    context = ExecutionContext()
    
    # 2. Plan
    objective = f"Analyze the architecture of the RAG system in {RAG_DIR}. List files first, then read the README.md to understand it."
    print(f"\n[2] Planning for: {objective}")
    
    available_tools = tool_manager.list_tools(limit=10)
    
    plan = planner.plan(objective, available_tools)
    print("Plan Generated.")
    
    # 3. Validate
    print("\n[3] Validating Plan...")
    validation_result = validator.validate(plan)
    if not validation_result.valid:
        print(f"Plan Validation Failed: {validation_result.errors}")
        return
    print("Plan is VALID.")
    
    # 4. Execute
    print("\n[4] Executing Plan...")
    
    context.set("input", objective) 
    
    execution_trace = []
    
    for step in plan.steps:
        print(f"  > Step {step.step_id}: {step.intent}...")
        try:
            result = await executor.execute_step(step, context)
            
            trace_item = {
                "step_id": step.step_id,
                "intent": step.intent,
                "status": result.status,
                "result": str(result.result)[:100] + "..." if result.result and len(str(result.result)) > 100 else result.result,
                "error": result.error
            }
            execution_trace.append(trace_item)
            
            if result.status != "success" and step.fallback_strategy == "fail":
                print(f"    Failed: {result.error}")
                break
            else:
                print(f"    Success.")
                
        except Exception as e:
            print(f"    Critical Execution Exception: {e}")
            execution_trace.append({"step_id": step.step_id, "status": "critical_fail", "error": str(e)})
            break
            
    # 5. Summary Report
    print("\n" + "="*40)
    print("       EXECUTION SUMMARY")
    print("="*40)
    
    print("\n[Plan Overview]")
    for step in plan.steps:
        print(f"- {step.step_id}: {step.intent} (Op: {step.operation.type.value})")
        
    print("\n[Execution Trace]")
    for trace in execution_trace:
        status_icon = "✅" if trace['status'] == 'success' else "❌"
        print(f"{status_icon} {trace['step_id']}: {trace['result']}")
        if trace['error']:
            print(f"   Error: {trace['error']}")
            
    print("\n[Final Context State]")
    for key in context.variables.keys():
        val = context.variables[key]
        val_preview = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
        print(f"- {key}: {val_preview}")
        
    print("="*40)

if __name__ == "__main__":
    asyncio.run(run_integration_test())
