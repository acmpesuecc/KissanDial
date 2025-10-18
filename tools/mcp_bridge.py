"""
MCP Bridge for KissanDial
Converts MCP tools to LlamaIndex FunctionTool on-the-fly
"""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

try:
    from llama_index.core.tools import FunctionTool, ToolMetadata
except ImportError:
    print("LlamaIndex not available, using mock classes", file=sys.stderr)
    
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

# Import MCP clients with fallback
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("MCP not available, using mock implementations", file=sys.stderr)
    MCP_AVAILABLE = False
    
    class ClientSession:
        def __init__(self, *args, **kwargs):
            pass
        async def initialize(self): pass
        async def list_tools(self): return type('obj', (object,), {'tools': []})()
        async def call_tool(self, name, args): 
            return type('obj', (object,), {'content': [type('obj', (object,), {'text': 'Mock response'})()]})()
        async def close(self): pass
    
    def stdio_client(process):
        return None, None

class MCPBridge:
    """Bridge between MCP servers and LlamaIndex tools"""
    
    def __init__(self):
        self.active_sessions: Dict[str, ClientSession] = {}
        self.tools: List[FunctionTool] = []
        
    async def connect_stdio_server(self, server_name: str, server_path: str) -> bool:
        """Connect to an MCP server via stdio"""
        try:
            if not MCP_AVAILABLE:
                print(f"MCP not available, cannot connect to {server_name}", file=sys.stderr)
                return False
                
            # Start the server process
            process = await asyncio.create_subprocess_exec(
                sys.executable, server_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Create stdio client session
            read_stream, write_stream = await stdio_client(process.stdout, process.stdin)
            session = ClientSession(read_stream, write_stream)
            
            # Initialize the session
            await session.initialize()
            
            # Store the session
            self.active_sessions[server_name] = session
            
            print(f"Connected to MCP server: {server_name}", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"Failed to connect to MCP server {server_name}: {e}", file=sys.stderr)
            return False
    
    async def load_tools_from_server(self, server_name: str) -> List[FunctionTool]:
        """Load tools from a connected MCP server"""
        if server_name not in self.active_sessions:
            print(f"Server {server_name} not connected", file=sys.stderr)
            return []
        
        session = self.active_sessions[server_name]
        tools = []
        
        try:
            # List available tools from the server
            response = await session.list_tools()
            
            for tool_def in response.tools:
                # Create a LlamaIndex FunctionTool for each MCP tool
                llamaindex_tool = self._create_llamaindex_tool(
                    server_name, 
                    tool_def, 
                    session
                )
                tools.append(llamaindex_tool)
                
            print(f"Loaded {len(tools)} tools from {server_name}", file=sys.stderr)
            return tools
            
        except Exception as e:
            print(f"Error loading tools from {server_name}: {e}", file=sys.stderr)
            return []
    
    def _create_llamaindex_tool(self, server_name: str, tool_def: Any, session: ClientSession) -> FunctionTool:
        """Create a LlamaIndex FunctionTool from an MCP tool definition"""
        
        async def tool_function(**kwargs) -> str:
            """Async wrapper function that calls the MCP tool"""
            try:
                # Call the MCP tool
                response = await session.call_tool(tool_def.name, kwargs)
                
                # Extract text content from response
                result_texts = []
                for content in response.content:
                    if hasattr(content, 'text'):
                        result_texts.append(content.text)
                    elif hasattr(content, 'data'):
                        result_texts.append(str(content.data))
                    else:
                        result_texts.append(str(content))
                
                return "\n".join(result_texts) if result_texts else "No response from tool"
                
            except Exception as e:
                return f"Error calling MCP tool {tool_def.name}: {str(e)}"
        
        def sync_tool_function(**kwargs) -> str:
            """Synchronous wrapper for the async tool function"""
            try:
                # Run the async function in the current event loop or create a new one
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we need to handle this differently
                    # For now, we'll create a new task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, tool_function(**kwargs))
                        return future.result()
                else:
                    return loop.run_until_complete(tool_function(**kwargs))
            except Exception as e:
                return f"Error executing tool {tool_def.name}: {str(e)}"
        
        # Create the tool metadata
        metadata = ToolMetadata(
            name=f"{server_name}_{tool_def.name}",
            description=f"[{server_name}] {tool_def.description}",
        )
        
        # Create the FunctionTool
        return FunctionTool.from_defaults(
            fn=sync_tool_function,
            name=f"{server_name}_{tool_def.name}",
            description=f"[{server_name}] {tool_def.description}",
        )
    
    async def load_all_mcp_tools(self, server_configs: Dict[str, str]) -> List[FunctionTool]:
        """Load tools from all configured MCP servers"""
        all_tools = []
        
        for server_name, server_path in server_configs.items():
            # Connect to the server
            connected = await self.connect_stdio_server(server_name, server_path)
            
            if connected:
                # Load tools from the server
                server_tools = await self.load_tools_from_server(server_name)
                all_tools.extend(server_tools)
        
        self.tools = all_tools
        return all_tools
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for server_name, session in self.active_sessions.items():
            try:
                await session.close()
                print(f"Disconnected from {server_name}", file=sys.stderr)
            except Exception as e:
                print(f"Error disconnecting from {server_name}: {e}", file=sys.stderr)
        
        self.active_sessions.clear()
        self.tools.clear()


def load_mcp_tools() -> List[FunctionTool]:
    """
    Synchronous function to load MCP tools for use in LlamaIndex agent
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Configure MCP servers
    server_configs = {
        "subsidy": str(project_root / "servers" / "subsidy_mcp.py"),
        "price": str(project_root / "servers" / "price_mcp.py"),
        "weather": str(project_root / "servers" / "community" / "weather_mcp.py"),
    }
    
    # Create the bridge and load tools
    bridge = MCPBridge()
    
    try:
        # Run the async loading function
        tools = asyncio.run(bridge.load_all_mcp_tools(server_configs))
        print(f"Successfully loaded {len(tools)} MCP tools", file=sys.stderr)
        return tools
    except Exception as e:
        print(f"Error loading MCP tools: {e}", file=sys.stderr)
        return []


async def test_mcp_tools():
    """Test function for MCP tools"""
    bridge = MCPBridge()
    project_root = Path(__file__).parent.parent
    
    server_configs = {
        "subsidy": str(project_root / "servers" / "subsidy_mcp.py"),
        "price": str(project_root / "servers" / "price_mcp.py"),
    }
    
    try:
        tools = await bridge.load_all_mcp_tools(server_configs)
        
        print(f"Loaded {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.metadata.name}: {tool.metadata.description}")
        
        # Test a tool call
        if tools:
            print("\nTesting subsidy search...")
            for tool in tools:
                if "subsidy_search" in tool.metadata.name:
                    result = tool.call(query="tractor")
                    print(f"Result: {result}")
                    break
        
        await bridge.disconnect_all()
        
    except Exception as e:
        print(f"Test error: {e}")


if __name__ == "__main__":
    asyncio.run(test_mcp_tools())
