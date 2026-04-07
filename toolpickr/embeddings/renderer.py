# Render the tool defination into text for the embedding model to consume

import json
from toolpickr.core.tool import ToolDefinition

def render_tool_text(tool: ToolDefinition) -> str:
    """Converts a ToolDefinition into a formatted text string for embedding."""
    # Convert the parameters dict to a JSON string for readability
    params_str = json.dumps(tool.parameters) if tool.parameters else "{}"
    
    # This is what the embedding model will "read".
    text = f"""
            Tool Name: {tool.name}
            Description: {tool.description}
            """
            #Parameters: {params_str}
            #Returns: {tool.returns or 'None'}
            #"""
    
    # Optionally append the category if it exists
    if tool.category:
        text += f"\nCategory: {tool.category}"
        
    return text

