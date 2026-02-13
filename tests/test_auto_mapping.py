import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.core.agent import LangGraphAgent
from agent.utils.logger import logger

async def run_test():
    print(f"\n{'='*50}")
    print(f"TEST: New Skill Auto-Mapping")
    print(f"QUERY: Resize photo.jpg to 100x100")
    print(f"{'='*50}")
    
    agent = LangGraphAgent()
    
    # Check if resize_image is loaded
    # tools = agent.app.nodes['planner'].tools.list_tools() # Hack to access tools via graph nodes? 
    # No, easy way is to check logs or just run.
    
    query = "Resize photo.jpg to 100x100"
    
    try:
        result = await agent.ainvoke(query)
        
        print("\n[RESULT]")
        print(f"Status: {result.get('status')}")
        print(f"Response: {result.get('response')}")
        
        trace = result.get('trace')
        if trace:
            past_steps = trace.get('past_steps', [])
            for step in past_steps:
                print(f"  - Step: {step.step_id}")
                print(f"    Tool: {step.meta.get('tool')}")
                print(f"    Args: {step.meta.get('args')}")
                print(f"    Result: {step.result}")
                
                if step.meta.get('tool') == 'resize_image':
                    args = step.meta.get('args', {})
                    if args.get('img_path') == 'photo.jpg' and args.get('width') == 100:
                         print("    [VERIFICATION] SUCCESS: Auto-mapping worked!")
                    else:
                         print("    [VERIFICATION] FAILED: Args mismatch")

    except Exception as e:
        print(f"\n[EXCEPTION] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_test())
