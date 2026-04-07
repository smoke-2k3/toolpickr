from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ToolDefinition(BaseModel):
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="A dict representing the JSON schema of the function inputs")
    returns: Optional[str] = Field(default=None, description="The return value of the function")
    category: Optional[str] = Field(default=None, description="The category of the tool")

    def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None, **data):
        # If parameters is not provided, use an empty dict
        if parameters is None:
            parameters = {}
        # Call the parent constructor with keyword arguments
        super().__init__(name=name, description=description, parameters=parameters, **data)
