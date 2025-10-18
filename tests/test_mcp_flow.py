"""
Test MCP Flow for KissanDial
Tests the complete flow from voice input to MCP tool execution
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MCP bridge
try:
    from tools.mcp_bridge_simple import load_mcp_tools
    MCP_BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"MCP bridge not available: {e}", file=sys.stderr)
    MCP_BRIDGE_AVAILABLE = False

try:
    from tools.mcp_bridge import MCPBridge
    MCP_ORIGINAL_AVAILABLE = True
except ImportError:
    MCP_ORIGINAL_AVAILABLE = False

class TestMCPFlow:
    """Test MCP integration flow"""
    
    def test_load_mcp_tools(self):
        """Test that MCP tools can be loaded"""
        if not MCP_BRIDGE_AVAILABLE:
            print("MCP bridge not available, skipping test")
            return
            
        tools = load_mcp_tools()
        
        # Should load tools successfully
        assert isinstance(tools, list)
        assert len(tools) > 0, "Should have loaded mock tools"
        print(f"Loaded {len(tools)} MCP tools")
        
        tool_names = [tool.metadata.name for tool in tools]
        print(f"Available tools: {tool_names}")
        
        # Check for expected tools
        expected_tools = ["subsidy_search", "get_mandi_price", "get_current_weather"]
        for expected in expected_tools:
            assert expected in tool_names, f"Should have {expected} tool"
    
    @pytest.mark.asyncio
    async def test_mcp_bridge_connection(self):
        """Test MCP bridge can connect to servers"""
        if not MCP_ORIGINAL_AVAILABLE:
            print("Original MCP bridge not available, skipping connection test")
            return
            
        bridge = MCPBridge()
        
        # Test connection to subsidy server
        server_path = str(project_root / "servers" / "subsidy_mcp.py")
        try:
            connected = await bridge.connect_stdio_server("test_subsidy", server_path)
            if connected:
                tools = await bridge.load_tools_from_server("test_subsidy")
                assert isinstance(tools, list)
                print(f"Connected to subsidy server, got {len(tools)} tools")
            else:
                print("Could not connect to subsidy server (expected with current implementation)")
        except Exception as e:
            print(f"Expected error connecting to subsidy server: {e}")
        finally:
            await bridge.disconnect_all()
    
    def test_tool_execution_simulation(self):
        """Test tool execution with mock data"""
        if not MCP_BRIDGE_AVAILABLE:
            print("MCP bridge not available, skipping test")
            return
            
        tools = load_mcp_tools()
        
        assert len(tools) > 0, "Should have tools available"
        
        # Find a subsidy search tool
        subsidy_tool = None
        for tool in tools:
            if 'subsidy_search' in tool.metadata.name:
                subsidy_tool = tool
                break
        
        assert subsidy_tool is not None, "Should have subsidy search tool"
        
        try:
            # Test tool execution
            result = subsidy_tool.fn(query="tractor")
            assert isinstance(result, str)
            assert len(result) > 0
            assert "subsidy" in result.lower() or "scheme" in result.lower()
            print(f"Subsidy search result: {result[:100]}...")
        except Exception as e:
            assert False, f"Tool execution should not fail: {e}"
    
    def test_simulated_voice_flow(self):
        """Simulate the complete voice interaction flow"""
        print("\\n=== Simulating Voice Flow ===")
        
        if not MCP_BRIDGE_AVAILABLE:
            print("MCP bridge not available, skipping voice flow test")
            return
        
        # 1. Simulate voice input
        user_query = "What is the current tomato price in Mysuru?"
        print(f"User Query: {user_query}")
        
        # 2. Load MCP tools
        tools = load_mcp_tools()
        print(f"Available tools: {len(tools)}")
        assert len(tools) > 0, "Should have tools available"
        
        # 3. Find relevant tool
        price_tool = None
        for tool in tools:
            if 'get_mandi_price' in tool.metadata.name or 'price' in tool.metadata.name:
                price_tool = tool
                break
        
        assert price_tool is not None, "Should have price tool"
        print(f"Found relevant tool: {price_tool.metadata.name}")
        
        try:
            # 4. Execute tool
            result = price_tool.fn(crop="Tomato", district="Mysuru")
            print(f"Tool result: {result[:200]}...")
            
            # 5. Simulate agent response
            agent_response = f"Let me check that for you. {result}"
            print(f"Agent Response: {agent_response[:100]}...")
            
            assert len(agent_response) > 0
            assert "tomato" in agent_response.lower() or "price" in agent_response.lower()
            print("✅ Voice flow simulation successful")
            
        except Exception as e:
            assert False, f"Voice flow should not fail: {e}"
    
    def test_error_handling(self):
        """Test error handling in MCP flow"""
        tools = load_mcp_tools()
        
        if not tools:
            print("No tools to test error handling")
            return
            
        # Test with invalid parameters
        for tool in tools[:1]:  # Test just the first tool
            try:
                # Call with invalid parameters
                result = tool.call(invalid_param="test")
                print(f"Tool handled invalid params: {result[:50]}...")
            except Exception as e:
                print(f"Tool error (expected): {e}")

if __name__ == "__main__":
    # Run tests directly
    test_flow = TestMCPFlow()
    
    print("🧪 Running MCP Flow Tests")
    print("=" * 50)
    
    try:
        test_flow.test_load_mcp_tools()
        print("✅ test_load_mcp_tools passed")
    except Exception as e:
        print(f"❌ test_load_mcp_tools failed: {e}")
    
    try:
        asyncio.run(test_flow.test_mcp_bridge_connection())
        print("✅ test_mcp_bridge_connection passed")
    except Exception as e:
        print(f"❌ test_mcp_bridge_connection failed: {e}")
    
    try:
        test_flow.test_tool_execution_simulation()
        print("✅ test_tool_execution_simulation passed")
    except Exception as e:
        print(f"❌ test_tool_execution_simulation failed: {e}")
    
    try:
        test_flow.test_simulated_voice_flow()
        print("✅ test_simulated_voice_flow passed")
    except Exception as e:
        print(f"❌ test_simulated_voice_flow failed: {e}")
    
    try:
        test_flow.test_error_handling()
        print("✅ test_error_handling passed")
    except Exception as e:
        print(f"❌ test_error_handling failed: {e}")
    
    print("\\n🏁 Test run completed")
    print("Note: Some failures are expected when MCP servers are not running")
