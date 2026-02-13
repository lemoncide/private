import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.core.agent import LangGraphAgent
from agent.utils.logger import logger

async def run_test(name: str, query: str, runs: int = 5):
    print(f"\n{'='*50}")
    print(f"TEST: {name} ({runs} runs)")
    print(f"QUERY: {query}")
    print(f"{'='*50}")
    
    success_count = 0
    repair_count = 0
    
    for i in range(runs):
        print(f"\n--- Run {i+1}/{runs} ---")
        agent = LangGraphAgent()
        
        try:
            result = await agent.ainvoke(query)
            
            trace = result.get('trace')
            # Check for trace being None or a dict
            if isinstance(trace, dict):
                 past_steps = trace.get('past_steps', [])
                 
                 # Check if repair was triggered
                 repaired = any(step.status == "failed" for step in past_steps)
                 if repaired:
                     repair_count += 1
            else:
                 past_steps = []
                 repaired = False
                
            status = result.get('status')
            if status == "completed":
                success_count += 1
                print(f"Result: SUCCESS (Repaired: {repaired})")
                print(f"Response: {result.get('response')}")
            else:
                print(f"Result: FAILED (Status: {status})")
                if isinstance(trace, dict) and trace.get('error'):
                    print(f"Error: {trace.get('error')}")
            
        except Exception as e:
            print(f"EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[SUMMARY] {name}")
    try:
        print(f"Success Rate: {success_count}/{runs} ({success_count/runs*100:.1f}%)")
        print(f"Repair Rate: {repair_count}/{runs} ({repair_count/runs*100:.1f}%)")
    except:
        pass

async def main():
    # Test 1: Math (Simple Computation)
    # await run_test("Math Computation", "Calculate 50 * 3", runs=1)
    
    # Test 2: File System (Side Effect)
    # Ensure file is clean
    test_file = "test_graph_output.txt"
    if os.path.exists(test_file):
        os.remove(test_file)
        
    await run_test("File Write", f"Write 'Hello LangGraph' to {test_file}", runs=1)
    
    # Test 3: Search (Reasoning & Tool Use)
    await run_test("Search & Summarize", "Search for 'LangGraph' and summarize what it is", runs=1)

if __name__ == "__main__":
    asyncio.run(main())
