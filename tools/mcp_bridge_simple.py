"""
Simplified MCP Bridge for KissanDial
Creates mock tools that simulate MCP functionality for now
"""

import sys
from typing import List
from pathlib import Path

try:
    from llama_index.core.tools import FunctionTool, ToolMetadata
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    print("LlamaIndex not available, using mock classes", file=sys.stderr)
    LLAMAINDEX_AVAILABLE = False
    
    class ToolMetadata:
        def __init__(self, name, description):
            self.name = name
            self.description = description
    
    class FunctionTool:
        def __init__(self, fn, metadata):
            self.fn = fn
            self.metadata = metadata
        
        @classmethod
        def from_defaults(cls, fn, name, description):
            metadata = ToolMetadata(name=name, description=description)
            return cls(fn, metadata)
        
        def call(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

def create_mock_subsidy_tool() -> FunctionTool:
    """Create a mock subsidy search tool"""
    
    def subsidy_search(query: str = "") -> str:
        """Mock subsidy search function"""
        mock_results = [
            {
                "scheme": "PM-KISAN Samman Nidhi",
                "description": "Direct income support to farmers",
                "amount": "₹6,000 per year",
                "eligibility": "Small and marginal farmers"
            },
            {
                "scheme": "Pradhan Mantri Fasal Bima Yojana",
                "description": "Crop insurance scheme",
                "amount": "Premium support up to 2%",
                "eligibility": "All farmers"
            },
            {
                "scheme": "Kisan Credit Card",
                "description": "Credit facility for agricultural needs",
                "amount": "Based on crop area and pattern",
                "eligibility": "All farmers with land records"
            }
        ]
        
        result = f"Subsidy search results for '{query}':\\n\\n"
        
        for i, subsidy in enumerate(mock_results, 1):
            if query.lower() in subsidy["scheme"].lower() or query.lower() in subsidy["description"].lower() or not query:
                result += f"{i}. {subsidy['scheme']}\\n"
                result += f"   Description: {subsidy['description']}\\n"
                result += f"   Amount: {subsidy['amount']}\\n"
                result += f"   Eligibility: {subsidy['eligibility']}\\n\\n"
        
        result += "Note: This is mock data. Real MCP servers will provide live subsidy information."
        return result
    
    return FunctionTool.from_defaults(
        fn=subsidy_search,
        name="subsidy_search",
        description="Search for government subsidies for farmers based on query"
    )

def create_mock_price_tool() -> FunctionTool:
    """Create a mock price lookup tool"""
    
    def get_mandi_price(crop: str = "", district: str = "Mysuru") -> str:
        """Mock price lookup function"""
        import random
        
        mock_prices = {
            "rice": 2500,
            "wheat": 2100,
            "tomato": 1800,
            "onion": 1200,
            "potato": 1500,
            "cotton": 5800,
            "sugarcane": 3200
        }
        
        crop_lower = crop.lower()
        base_price = mock_prices.get(crop_lower, random.randint(1500, 4000))
        
        # Add some variation
        current_price = base_price + random.randint(-200, 200)
        
        result = f"Market Price for {crop.title()} in {district}:\\n\\n"
        result += f"Current Price: ₹{current_price}/quintal\\n"
        result += f"Previous Day: ₹{base_price}/quintal\\n"
        
        change = current_price - base_price
        if change > 0:
            result += f"Change: +₹{change} (↗️)\\n"
        elif change < 0:
            result += f"Change: ₹{change} (↘️)\\n"
        else:
            result += "Change: No change (➡️)\\n"
        
        result += f"\\nMarket: {district} Mandi\\n"
        result += f"Last Updated: Today\\n\\n"
        result += "Note: This is mock data. Real MCP servers will provide live market prices."
        
        return result
    
    return FunctionTool.from_defaults(
        fn=get_mandi_price,
        name="get_mandi_price", 
        description="Get current market price for crops in specific districts"
    )

def create_mock_weather_tool() -> FunctionTool:
    """Create a mock weather tool"""
    
    def get_current_weather(location: str = "Mysuru") -> str:
        """Mock weather function"""
        import random
        
        temp = random.randint(22, 32)
        humidity = random.randint(65, 85)
        conditions = ["Clear", "Partly Cloudy", "Cloudy", "Light Rain"]
        condition = random.choice(conditions)
        
        result = f"Current Weather in {location}:\\n\\n"
        result += f"Temperature: {temp}°C\\n"
        result += f"Condition: {condition}\\n"
        result += f"Humidity: {humidity}%\\n"
        result += f"Wind: {random.randint(5, 15)} km/h\\n\\n"
        
        # Agricultural advice
        if temp > 30:
            result += "Agricultural Advisory: High temperature - ensure adequate irrigation.\\n"
        if humidity > 80:
            result += "Agricultural Advisory: High humidity - monitor for diseases.\\n"
        if condition == "Light Rain":
            result += "Agricultural Advisory: Light rain expected - good for crops.\\n"
        
        result += "\\nNote: This is mock weather data. Real MCP servers will provide live weather information."
        
        return result
    
    return FunctionTool.from_defaults(
        fn=get_current_weather,
        name="get_current_weather",
        description="Get current weather conditions for agricultural planning"
    )

def load_mcp_tools() -> List[FunctionTool]:
    """
    Load MCP tools - currently returns mock tools
    In the future, this will connect to real MCP servers
    """
    print("Loading MCP tools (mock implementation)", file=sys.stderr)
    
    tools = []
    
    try:
        # Create mock tools for now
        tools.append(create_mock_subsidy_tool())
        tools.append(create_mock_price_tool())
        tools.append(create_mock_weather_tool())
        
        print(f"Successfully loaded {len(tools)} mock MCP tools", file=sys.stderr)
        
    except Exception as e:
        print(f"Error creating mock tools: {e}", file=sys.stderr)
    
    return tools

if __name__ == "__main__":
    # Test the mock tools
    print("Testing Mock MCP Tools")
    print("=" * 30)
    
    tools = load_mcp_tools()
    
    for tool in tools:
        print(f"\\nTesting {tool.metadata.name}:")
        print("-" * 20)
        
        if tool.metadata.name == "subsidy_search":
            result = tool.fn(query="tractor")
        elif tool.metadata.name == "get_mandi_price":
            result = tool.fn(crop="tomato", district="Mysuru")
        elif tool.metadata.name == "get_current_weather":
            result = tool.fn(location="Bangalore")
        else:
            result = tool.fn()
            
        # Convert result to string if it's not already
        result_str = str(result)
        print(result_str[:200] + "..." if len(result_str) > 200 else result_str)
    
    print("\\n✅ Mock MCP tools are working!")
