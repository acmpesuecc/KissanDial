from flask import Flask, request, session
from twilio.twiml.voice_response import VoiceResponse, Gather
import os
import pandas as pd
from twilio.rest import Client

# Try to import LlamaIndex components, with fallbacks for testing
try:
    from llama_index import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.llms import OpenAI
    from llama_index.tools import QueryEngineTool, ToolMetadata, FunctionTool
    from llama_index.agent.openai.base import OpenAIAgentWorker
    from llama_index.memory import ChatMemoryBuffer
    LLAMA_INDEX_AVAILABLE = True
except ImportError as e:
    print(f"LlamaIndex import error: {e}")
    print("Running in test mode without LlamaIndex")
    LLAMA_INDEX_AVAILABLE = False

# Create mock classes for testing (always available)
class SimpleDirectoryReader:
    def __init__(self, input_files):
        self.input_files = input_files
    def load_data(self):
        return []

class VectorStoreIndex:
    @classmethod
    def from_documents(cls, docs):
        return cls()
    def as_query_engine(self, similarity_top_k=6):
        return MockQueryEngine()

class MockQueryEngine:
    def query(self, query):
        return f"Mock response for: {query}"

class OpenAI:
    def __init__(self, temperature=0, model="gpt-4"):
        self.temperature = temperature
        self.model = model

class QueryEngineTool:
    def __init__(self, query_engine, metadata):
        self.query_engine = query_engine
        self.metadata = metadata

class ToolMetadata:
    def __init__(self, name, description):
        self.name = name
        self.description = description

class FunctionTool:
    @classmethod
    def from_defaults(cls, fn, name, description):
        return cls()

class OpenAIAgentWorker:
    @classmethod
    def from_tools(cls, tools, system_prompt, memory, llm, verbose=True, allow_parallel_tool_calls=False):
        return cls()
    def as_agent(self):
        return MockAgent()

class ChatMemoryBuffer:
    @classmethod
    def from_defaults(cls, token_limit=2048):
        return cls()

class MockAgent:
    def chat(self, message):
        return f"Mock agent response to: {message}"


# Set up OpenAI API key - use dummy key for testing if not set
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"

to_say = 'Hi welcome to the Agricultural Assistant. How can I help you today?'

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace-me")


# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "data", "subsidies", "central", "main_subsidy_data.csv")

# Initialize components conditionally
if LLAMA_INDEX_AVAILABLE:
    try:
        # Load and index documents
        subsidy_docs = SimpleDirectoryReader(input_files=[csv_path]).load_data()
        subsidy_index = VectorStoreIndex.from_documents(subsidy_docs)
        # Create query engine
        subsidy_engine = subsidy_index.as_query_engine(similarity_top_k=6)
    except Exception as e:
        print(f"Error initializing LlamaIndex components: {e}")
        print("Falling back to mock components")
        LLAMA_INDEX_AVAILABLE = False
        subsidy_engine = MockQueryEngine()
else:
    subsidy_engine = MockQueryEngine()

# Load the CSV file
df = pd.read_csv(csv_path)

def send_sms_with_subsidy_info(query: str) -> str:
    """
    Send SMS with 'how_to_apply' information for relevant subsidies.
    """
    print("INSIDE SEND SMS FUNCTION")
    print(f"Query: {query}")
    results = subsidy_engine.query(query)
    print(f"Results: {results}")
    
    sms_body = "How to apply for relevant subsidies:\n\n"
    # for subsidy in relevant_subsidies:
    sms_body += f"{results}"

    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)
    
    try:
        message = client.messages.create(
            from_='+17206276852',
            body=sms_body,
            to="+919108006252"
        )
        return f"SMS sent successfully with SID: {message.sid}"
    except Exception as e:
        return f"Error sending SMS: {str(e)}"

# Define tools
subsidy_tool = QueryEngineTool(
    query_engine=subsidy_engine,
    metadata=ToolMetadata(
        name="get_subsidy_info",
        description="Provides information about subsidies for farmers in India.",
    ),
)

sms_tool = FunctionTool.from_defaults(
    fn=send_sms_with_subsidy_info,
    name="send_sms_with_subsidy_info",
    description="Sends an SMS with 'how_to_apply' information for relevant subsidies.",
)

# Custom prompt for the agent
CUSTOM_PROMPT = """
You are a helpful conversational assistant. Below are the global details about the usecase which you need to abide by strictly:
<global_details>
Task: You are an intelligent agricultural assistant. Your goal is to provide helpful information to farmers about subsidies, weather, and farming advice. Always ask follow-up questions to understand the farmer's specific needs better before giving detailed information.
Use the provided tools to gather information, but always ask follow-up questions before providing detailed answers. This will help you give more targeted and useful information to the farmer. For eg: "Agent: Are you looking for subsidies related to seeds, crops, machines or insurance?"
Response style: Your responses must be very short, concise and helpful.
</global_details>

You are currently in a specific state of a conversational flow which is described below. This state is one amongst many other states which constitutes the entire conversation design of the interaction between a user and you as the assistant. Details about the current state:
<state_details>
Name: Help farmer
Goal: To answer all queries regarding government subsidies for farmers and weather.
Instructions: 1. Greet the user and understand their question. 2. If the user asks about subsidies always ask them if they are looking for subsidies for crops, seeds, machines or insurance if this information is not already known. If the user asks about the weather, call TOOL: get_weather_info 3. Answer user's questions using fetched data.
Tools: ["get_subsidy_info", "get_weather_info", "send_sms_with_subsidy_info"]
</state_details>

Remember to follow the below rules strictly:
1. Ensure coherence in the conversation. Responses should engage the user and maintain a flow that feels like a natural dialogue.
2. Use only the available tools in the bot's response.
3. Avoid generating any Unicode or Hindi characters.
4. Do not address the user and refrain from using any name for the user.
5. Strictly refrain from using emojis in your responses.
6. Your responses should be engaging, short and crisp. It should be more human conversation like.

- Use informal, more conversational and colloquial language. Avoid decorative words and choice of too much drama in your language.
- Avoid bulleted lists, markdowns, structured outputs, and any special characters like double quotes, single quotes, or hyphen etc in your responses.
- Avoid any numericals or SI units in your outputs. Ex: if you want to say 12:25, say twelve twenty five, or if you want to say 100m^2, say hundred meter square since this interaction is over call. Other fields can have numericals.
- Avoid any emoji or smiley faces since this interaction is over call.
- Call relevant tools whether it be some api or a retrieval tool to fetch context needed to answer any query that the user might have. First decide if a tool call is needed in the thought and then call the appropriate tool. Respond to the user with a variant of 'let me check that for you' and then call the tool in the same turn.
"""

# Initialize the agent conditionally
if LLAMA_INDEX_AVAILABLE:
    try:
        llm = OpenAI(temperature=0, model="gpt-4")
        memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
        agent_worker = OpenAIAgentWorker.from_tools(
          [subsidy_tool, sms_tool], system_prompt=CUSTOM_PROMPT, 
          memory=memory, llm=llm,
          verbose=True,
          allow_parallel_tool_calls=False,
        )
        agent = agent_worker.as_agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Falling back to mock agent")
        agent = MockAgent()
else:
    agent = MockAgent()

@app.route('/voice', methods=['POST'])
def voice():
    sid = request.form.get('CallSid')
    # Default greeting if not set yet for this session
    to_say = session.get(f'to_say_{sid}', "Hi welcome to the Agricultural Assistant. How can I help you today?")
    print(f'to_say {to_say}')
    resp = VoiceResponse()
    gather = Gather(input='speech', action='/handle-speech', method='POST', speechTimeout='1', speechModel='experimental_conversations', enhanced=True)
    gather.say(to_say)
    resp.append(gather)
    resp.redirect('/voice')
    return str(resp)


@app.route('/handle-speech', methods=['POST'])
def handlespeech():
    sid = request.form.get('CallSid')
    resp = VoiceResponse()
    speechresult = request.form.get('SpeechResult')
    if speechresult:
        agentresponse = agent.chat(speechresult)
        print(f'User: {speechresult}')
        print(f'Assistant: {agentresponse}')
        session[f'to_say_{sid}'] = str(agentresponse)
        resp.redirect('/voice')
    else:
        resp.say("I'm sorry, I didn't catch that. Could you please repeat?")
        resp.redirect('/voice')
    return str(resp)


if __name__ == "__main__":
    app.run(debug=True)
