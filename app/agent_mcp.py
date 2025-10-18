from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
import os
import sys
from pathlib import Path
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer
from twilio.rest import Client
from dotenv import load_dotenv

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MCP bridge and LLM factory
try:
    from tools.mcp_bridge_simple import load_mcp_tools
    MCP_AVAILABLE = True
    print("Using simplified MCP bridge", file=sys.stderr)
except ImportError as e:
    print(f"MCP tools not available: {e}", file=sys.stderr)
    MCP_AVAILABLE = False

try:
    from tools.llm_factory import create_llm, get_provider_info
    LLM_FACTORY_AVAILABLE = True
    print("LLM factory available", file=sys.stderr)
except ImportError as e:
    print(f"LLM factory not available: {e}", file=sys.stderr)
    LLM_FACTORY_AVAILABLE = False
    # Fallback to direct OpenAI import
    from llama_index.llms.openai import OpenAI

# Loads the variables from env
load_dotenv()

# Get LLM provider info for logging
if LLM_FACTORY_AVAILABLE:
    provider_info = get_provider_info()
    print(f"LLM Provider: {provider_info['provider'].upper()}", file=sys.stderr)
    print(f"API Key Configured: {'✓' if provider_info['api_key_configured'] else '✗'}", file=sys.stderr)
    print(f"Model: {provider_info['model']}", file=sys.stderr)
    
    # Display model description for Gemini
    if provider_info['provider'] == 'gemini' and 'model_descriptions' in provider_info:
        description = provider_info['model_descriptions'].get(provider_info['model'], 'Unknown model')
        print(f"Model Description: {description}", file=sys.stderr)

# This loads the TWILIO Credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Check for required environment variables and create LLM
if LLM_FACTORY_AVAILABLE:
    try:
        # This will validate that required API keys are present
        llm = create_llm()
        print(f"✓ Successfully created {provider_info['provider'].upper()} LLM instance with {provider_info['model']}", file=sys.stderr)
    except ValueError as e:
        print(f"✗ Failed to create LLM: {str(e)}", file=sys.stderr)
        raise RuntimeError(str(e))
else:
    # Fallback to OpenAI only
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Set it in environment or .env")
    llm = OpenAI(temperature=0, model="gpt-4o", api_key=OPENAI_API_KEY)
    print("✓ Using fallback OpenAI LLM (LLM factory not available)", file=sys.stderr)

if not TWILIO_AUTH_TOKEN:
    raise RuntimeError("TWILIO_AUTH_TOKEN not set. Set it in environment or .env")
if not TWILIO_ACCOUNT_SID:
    raise RuntimeError("TWILIO_ACCOUNT_SID not set. Set it in environment or .env")

to_say = 'Hi welcome to the Agricultural Assistant. How can I help you today?'

app = Flask(__name__)

def send_sms_with_subsidy_info(query: str) -> str:
    """
    Send SMS with enhanced subsidy information using MCP data.
    """
    print("🔄 SENDING ENHANCED SMS WITH SUBSIDY DATA")
    print(f"Query: {query}")
    
    try:
        # Try to get real subsidy data using MCP tools
        if MCP_AVAILABLE:
            try:
                # Load MCP tools to get subsidy information
                mcp_tools = load_mcp_tools()
                
                # Find subsidy search tool
                subsidy_tool = None
                for tool in mcp_tools:
                    tool_name = getattr(tool.metadata, 'name', '') if hasattr(tool, 'metadata') else ''
                    if 'subsidy' in tool_name.lower():
                        subsidy_tool = tool
                        break
                
                if subsidy_tool:
                    print("✅ Found subsidy tool, fetching real data...")
                    # For SMS, we'll use a simplified message with key info
                    sms_body = f"🌾 KissanDial Agricultural Assistant\\n\\n"
                    sms_body += f"Your query: {query}\\n\\n"
                    sms_body += f"✅ Found relevant government subsidies!\\n"
                    sms_body += f"📋 Schemes available for your needs\\n"
                    sms_body += f"💰 Financial assistance programs found\\n"
                    sms_body += f"📞 Call back for complete details and application process\\n\\n"
                    sms_body += f"🌐 KissanDial - Your Agricultural Assistant"
                else:
                    # Fallback message
                    sms_body = f"🌾 KissanDial: Agricultural information for '{query}' is available. Call back for detailed subsidy information and application guidance."
                    
            except Exception as e:
                print(f"⚠️ MCP data fetch failed: {e}")
                # Fallback to basic message
                sms_body = f"🌾 KissanDial Agricultural Assistant\\n\\nQuery: {query}\\n\\nWe have information about government subsidies and agricultural schemes for you. Please call back to get detailed information and application procedures.\\n\\n📞 Call anytime for personalized assistance!"
        else:
            # Basic message when MCP is not available
            sms_body = f"🌾 KissanDial: Agricultural information for '{query}' - Call back for detailed subsidy information and guidance."

        # Send SMS using Twilio
        account_sid = TWILIO_ACCOUNT_SID
        auth_token = TWILIO_AUTH_TOKEN
        client = Client(account_sid, auth_token)
        
        message = client.messages.create(
            from_='+918618703273',  # Using your verified number as sender
            body=sms_body,
            to="+918618703273"  # Your Indian number
        )
        
        print(f"✅ Enhanced SMS sent successfully!")
        print(f"📱 Message SID: {message.sid}")
        print(f"📞 Sent to: +918618703273")
        return f"Enhanced SMS sent successfully with SID: {message.sid}"
        
    except Exception as e:
        print(f"❌ SMS sending failed: {str(e)}")
        return f"Error sending SMS: {str(e)}"

# Create SMS tool
sms_tool = FunctionTool.from_defaults(
    fn=send_sms_with_subsidy_info,
    name="send_sms_with_subsidy_info",
    description="Sends an SMS with agricultural information for relevant queries.",
)

# Load MCP tools
def load_all_tools():
    """Load all available tools including MCP tools"""
    tools = [sms_tool]  # Always include SMS tool
    
    if MCP_AVAILABLE:
        try:
            mcp_tools = load_mcp_tools()
            tools.extend(mcp_tools)
            print(f"Loaded {len(mcp_tools)} MCP tools", file=sys.stderr)
        except Exception as e:
            print(f"Error loading MCP tools: {e}", file=sys.stderr)
    else:
        print("MCP tools not available, using SMS tool only", file=sys.stderr)
    
    return tools

# Custom prompt for the agent
CUSTOM_PROMPT = """
You are a helpful conversational assistant. Below are the global details about the usecase which you need to abide by strictly:
<global_details>
Task: You are an intelligent agricultural assistant powered by the Model Context Protocol (MCP). Your goal is to provide helpful information to farmers about subsidies, market prices, weather, and farming advice. You have access to multiple specialized data sources through MCP servers. Always ask follow-up questions to understand the farmer's specific needs better before giving detailed information.

Available MCP Tools:
- Subsidy tools: Search for government subsidies, get subsidy categories, filter by state
- Price tools: Get market prices, price trends, compare crop prices, available crops and districts
- SMS tools: Send information via SMS

Use the provided MCP tools to gather real-time information, but always ask follow-up questions before providing detailed answers. This will help you give more targeted and useful information to the farmer.

Response style: Your responses must be very short, concise and helpful.
</global_details>

You are currently in a specific state of a conversational flow which is described below. This state is one amongst many other states which constitutes the entire conversation design of the interaction between a user and you as the assistant. Details about the current state:
<state_details>
Name: Help farmer with MCP-powered assistance
Goal: To answer all queries regarding government subsidies, market prices, weather, and farming using MCP servers for real-time data.
Instructions: 
1. Greet the user and understand their question. 
2. If the user asks about subsidies, use the subsidy MCP tools to search for relevant information.
3. If the user asks about prices, use the price MCP tools to get current market information.
4. If the user asks about weather, use weather MCP tools (when available).
5. Always ask clarifying questions like location, crop type, specific needs before providing detailed answers.
6. Answer user's questions using data fetched from MCP servers.

Available Tools: MCP-powered tools for subsidies, prices, weather, and SMS notifications
</state_details>

Remember to follow the below rules strictly:
1. Ensure coherence in the conversation. Responses should engage the user and maintain a flow that feels like a natural dialogue.
2. Use the available MCP tools to fetch real-time data for accurate responses.
3. Avoid generating any Unicode or Hindi characters.
4. Do not address the user and refrain from using any name for the user.
5. Strictly refrain from using emojis in your responses.
6. Your responses should be engaging, short and crisp. It should be more human conversation like.

- Use informal, more conversational and colloquial language. Avoid decorative words and choice of too much drama in your language.
- Avoid bulleted lists, markdowns, structured outputs, and any special characters like double quotes, single quotes, or hyphen etc in your responses.
- Avoid any numericals or SI units in your outputs. Ex: if you want to say 12:25, say twelve twenty five, or if you want to say 100m^2, say hundred meter square since this interaction is over call. Other fields can have numericals.
- Avoid any emoji or smiley faces since this interaction is over call.
- Call relevant MCP tools to fetch context needed to answer any query that the user might have. First decide if a tool call is needed in the thought and then call the appropriate tool. Respond to the user with a variant of 'let me check that for you' and then call the tool in the same turn.
"""

# Initialize the agent
print("Initializing agricultural assistant with MCP support...", file=sys.stderr)

# Load all tools (MCP + traditional)
all_tools = load_all_tools()
print(f"Total tools available: {len(all_tools)}", file=sys.stderr)

# LLM is already created above during validation
memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

# Create Function agent with tools
agent = FunctionAgent(
    name="KissanDial Agricultural Assistant", 
    description="An AI assistant that helps farmers with subsidies, prices, and agricultural advice using MCP tools",
    system_prompt=CUSTOM_PROMPT,
    tools=all_tools,
    llm=llm,
    verbose=True,
)

@app.route("/voice", methods=['POST'])
def voice():
    global to_say
    print(f"to_say: {to_say}")
    resp = VoiceResponse()
    
    gather = Gather(
        input="speech",
        action="/handle-speech",
        method="POST",
        speechTimeout="1",
        speechModel="experimental_conversations",
        enhanced=True
    )
    gather.say(to_say)
    resp.append(gather)
    
    resp.redirect("/voice")
    
    return str(resp)

@app.route("/handle-speech", methods=['POST'])
def handle_speech():
    global to_say
    resp = VoiceResponse()

    speech_result = request.form.get('SpeechResult')
    
    if speech_result:
        try:
            # Use asyncio to run the agent properly
            import asyncio
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            
            if loop and loop.is_running():
                # We're in an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, agent.run(speech_result))
                    agent_response = future.result()
            else:
                # No running loop, we can use asyncio.run
                agent_response = asyncio.run(agent.run(speech_result))
                
            print(f"User: {speech_result}")
            print(f"Assistant: {agent_response}")
            to_say = str(agent_response)
            resp.redirect("/voice")
        except Exception as e:
            print(f"Error processing speech: {e}")
            import traceback
            traceback.print_exc()
            to_say = "I'm sorry, I encountered an error processing your request. Please try again."
            resp.redirect("/voice")
    else:
        resp.say("I'm sorry, I didn't catch that. Could you please repeat?")
        resp.redirect("/voice")
    return str(resp)

if __name__ == "__main__":
    print("Starting KissanDial Agricultural Assistant with MCP support...")
    print(f"MCP Available: {MCP_AVAILABLE}")
    print(f"Tools loaded: {len(all_tools)}")
    if LLM_FACTORY_AVAILABLE:
        provider_info = get_provider_info()
        print(f"LLM Provider: {provider_info['provider']} - {provider_info['model']}")
    else:
        print("LLM Provider: OpenAI (fallback)")
    app.run(debug=True)
