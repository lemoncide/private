from pydantic import BaseModel
from typing import Any, Dict

class BaseTool(BaseModel):
    name: str
    description: str

    def run(self, **kwargs) -> Any:
        raise NotImplementedError("Tool must implement run method")

class Calculator(BaseTool):
    name: str = "calculator"
    description: str = "Performs basic arithmetic operations."
    
    def run(self, **kwargs):
        print(f"Calculator run called with {kwargs}")
        return "success"

if __name__ == "__main__":
    try:
        print("Instantiating Calculator...")
        tool = Calculator()
        print(f"Tool instantiated: {tool}")
        
        print("Calling run...")
        result = tool.run(expression="5 * 3")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()
