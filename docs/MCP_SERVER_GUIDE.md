# KissanDial MCP Server Development Guide

## Overview

This guide explains how to add new MCP servers to KissanDial, expanding the agricultural data sources and capabilities.

## Quick Start - Adding a New MCP Server

### Step 1: Create Your MCP Server

Create a new file in the `servers/` directory:

```python
#!/usr/bin/env python3
"""
Your New MCP Server for KissanDial
Provides [your data type] through Model Context Protocol
"""

import asyncio
import json
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
from pydantic import BaseModel

# Define your data model
class YourDataQuery(BaseModel):
    query: str
    location: str = ""
    category: str = ""

# Initialize the server
server = Server("your-server-name")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools for your data source."""
    return [
        Tool(
            name="your_tool_name",
            description="Description of what your tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "location": {"type": "string", "description": "Location filter"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "your_tool_name":
        query = arguments.get("query", "")
        location = arguments.get("location", "")
        
        # Your data fetching logic here
        result = fetch_your_data(query, location)
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    else:
        raise ValueError(f"Unknown tool: {name}")

def fetch_your_data(query: str, location: str):
    """Implement your data fetching logic."""
    # Connect to your API/database
    # Process the query
    # Return structured data
    return {
        "results": [
            {
                "title": "Sample Data",
                "content": f"Data for {query} in {location}",
                "source": "Your Data Source"
            }
        ]
    }

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Update MCP Bridge

Add your server to the MCP bridge in `tools/mcp_bridge_simple.py`:

```python
def create_your_data_tool() -> FunctionTool:
    """Create your data search tool"""
    
    def your_data_search(query: str = "", location: str = "") -> str:
        """Your data search function"""
        # Mock data for testing
        results = {
            "data": f"Results for {query} in {location}",
            "source": "Your MCP Server"
        }
        return json.dumps(results, indent=2)
    
    return FunctionTool.from_defaults(
        fn=your_data_search,
        name="your_data_search",
        description="Search your data source based on query and location",
    )

# Add to load_mcp_tools function
def load_mcp_tools():
    tools = [
        create_mock_subsidy_tool(),
        create_mock_price_tool(), 
        create_mock_weather_tool(),
        create_your_data_tool()  # Add your tool here
    ]
    return tools
```

### Step 3: Configure Environment

Add your server to `.env`:

```bash
# MCP Server Configuration
MCP_SERVERS=stdio:./servers/subsidy_mcp.py,stdio:./servers/price_mcp.py,stdio:./servers/your_server.py
```

### Step 4: Test Your Server

```python
# Test your server
python servers/your_server.py

# Test integration
python tools/llm_test.py test
```

## MCP Server Types & Examples

### 1. Data API Servers

For servers that fetch data from external APIs:

```python
import httpx
import asyncio

async def fetch_api_data(query: str):
    """Fetch data from external API"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/search?q={query}")
        return response.json()
```

### 2. Database Servers

For servers that query local or remote databases:

```python
import sqlite3
import pandas as pd

def query_database(query: str):
    """Query local database"""
    conn = sqlite3.connect("data/your_data.db")
    df = pd.read_sql_query(
        "SELECT * FROM your_table WHERE content LIKE ?",
        conn, 
        params=(f"%{query}%",)
    )
    return df.to_dict('records')
```

### 3. File-based Servers

For servers that process local files:

```python
def search_files(query: str, data_dir: str = "data/"):
    """Search through local files"""
    results = []
    for file_path in Path(data_dir).glob("*.csv"):
        df = pd.read_csv(file_path)
        matches = df[df.astype(str).str.contains(query, case=False, na=False)]
        results.extend(matches.to_dict('records'))
    return results
```

## Real-World Examples

### Weather API Server

```python
# servers/weather_real_mcp.py
import httpx
from typing import Dict, Any

async def get_weather_data(location: str) -> Dict[str, Any]:
    """Get real weather data from OpenWeatherMap API"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params={
            "q": location,
            "appid": api_key,
            "units": "metric"
        })
        return response.json()
```

### Government Scheme API Server

```python
# servers/schemes_real_mcp.py
async def get_government_schemes(state: str, crop: str) -> Dict[str, Any]:
    """Get real government schemes from official API"""
    api_url = "https://api.india.gov.in/schemes"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, params={
            "state": state,
            "sector": "agriculture",
            "crop": crop
        })
        return response.json()
```

## Advanced Features

### 1. Caching

Add caching to improve performance:

```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

# Simple in-memory cache
_cache: Dict[str, tuple[Any, datetime]] = {}
CACHE_DURATION = timedelta(hours=1)

async def cached_fetch(cache_key: str, fetch_func):
    """Fetch data with caching"""
    now = datetime.now()
    
    if cache_key in _cache:
        data, timestamp = _cache[cache_key]
        if now - timestamp < CACHE_DURATION:
            return data
    
    # Fetch fresh data
    data = await fetch_func()
    _cache[cache_key] = (data, now)
    return data
```

### 2. Error Handling

Robust error handling for production:

```python
async def safe_api_call(url: str, params: dict):
    """Make API call with proper error handling"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        return {"error": "API timeout", "fallback": True}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}", "fallback": True}
    except Exception as e:
        return {"error": str(e), "fallback": True}
```

### 3. Data Validation

Validate input and output data:

```python
from pydantic import BaseModel, validator

class CropPriceQuery(BaseModel):
    crop: str
    district: str
    date_range: str = "7days"
    
    @validator('crop')
    def crop_must_be_valid(cls, v):
        valid_crops = ['rice', 'wheat', 'tomato', 'onion', 'potato']
        if v.lower() not in valid_crops:
            raise ValueError(f'Crop must be one of: {valid_crops}')
        return v.lower()
```

## Testing Your MCP Server

### Unit Tests

```python
# tests/test_your_server.py
import pytest
import asyncio
from servers.your_server import fetch_your_data

@pytest.mark.asyncio
async def test_data_fetching():
    result = await fetch_your_data("test query", "test location")
    assert "results" in result
    assert len(result["results"]) > 0

def test_data_validation():
    # Test your data validation logic
    assert validate_query("valid query") == True
    assert validate_query("") == False
```

### Integration Tests

```python
# Test with the actual agent
python -c "
import asyncio
from app.agent_mcp import agent

async def test_integration():
    response = await agent.run('Test query for your new data source')
    print(f'Response: {response}')

asyncio.run(test_integration())
"
```

## Deployment

### Docker Configuration

Add your server to Docker:

```dockerfile
# Dockerfile.mcp
FROM python:3.12-slim

WORKDIR /app
COPY servers/ ./servers/
COPY requirements.txt .

RUN pip install -r requirements.txt

# Add your server startup
CMD ["python", "servers/your_server.py"]
```

### Production Configuration

```yaml
# docker-compose.yml
services:
  your-mcp-server:
    build:
      dockerfile: Dockerfile.mcp
    environment:
      - YOUR_API_KEY=${YOUR_API_KEY}
    volumes:
      - ./data:/app/data
```

## Best Practices

### 1. **Data Source Standards**
- Always provide structured JSON output
- Include metadata (source, timestamp, confidence)
- Handle edge cases gracefully
- Implement proper error responses

### 2. **Performance Optimization**
- Use async/await for I/O operations
- Implement caching for frequently accessed data
- Set reasonable timeouts
- Batch requests when possible

### 3. **Security**
- Validate all input parameters
- Use environment variables for API keys
- Implement rate limiting
- Sanitize output data

### 4. **Monitoring**
- Log all API calls and responses
- Track performance metrics
- Monitor error rates
- Set up health checks

## Troubleshooting

### Common Issues

1. **Server Not Responding**
   ```bash
   # Test server directly
   python servers/your_server.py
   
   # Check logs
   tail -f logs/mcp_server.log
   ```

2. **Tool Not Loading**
   ```python
   # Verify tool registration
   from tools.mcp_bridge_simple import load_mcp_tools
   tools = load_mcp_tools()
   print([t.metadata.name for t in tools])
   ```

3. **API Connection Issues**
   ```python
   # Test API connectivity
   import httpx
   response = httpx.get("your-api-endpoint")
   print(response.status_code, response.text)
   ```

## Community & Resources

- **MCP Official Docs**: https://modelcontextprotocol.io/
- **KissanDial MCP Examples**: `servers/` directory
- **Testing Utilities**: `tools/llm_test.py`
- **Architecture Overview**: `docs/ARCHITECTURE.md`

---

## Summary

Adding new MCP servers to KissanDial is straightforward:

1. **Create** your server in `servers/your_server.py`
2. **Add** tool to `tools/mcp_bridge_simple.py`
3. **Configure** in `.env` file
4. **Test** with `python tools/llm_test.py`
5. **Deploy** with Docker if needed

Your new agricultural data source will automatically be available to the agent and accessible via voice calls and SMS!
