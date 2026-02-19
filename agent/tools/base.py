import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel

class BaseTool(BaseModel):
    name: str
    description: str
    args_schema: Optional[Dict[str, Any]] = None
    
    model_config = {'arbitrary_types_allowed': True}

    def run(self, **kwargs) -> Any:
        raise NotImplementedError("Tool must implement run method")

    async def arun(self, **kwargs) -> Any:
        return await asyncio.to_thread(self.run, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "description": self.description
        }
        if self.args_schema:
            data["args_schema"] = self.args_schema
        return data
