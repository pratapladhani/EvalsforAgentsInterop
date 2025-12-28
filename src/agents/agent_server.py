"""
Standalone Azure OpenAI Calendar Agent

A minimal FastAPI server providing an LLM-powered calendar scheduling agent.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import uvicorn
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class InvokeRequest(BaseModel):
    """Request model for agent invocation."""
    dataset_id: str
    test_case_id: str
    agent_id: str
    evaluation_run_id: str
    input: str


class ToolArgument(BaseModel):
    """Tool argument with name and value."""
    name: str
    value: Any


class ToolCall(BaseModel):
    """Tool call record."""
    name: str
    arguments: List[ToolArgument]
    response: Optional[Dict[str, Any]] = None  # MCP tool response


class InvokeResponse(BaseModel):
    """Response from agent invocation."""
    response: str
    tool_calls: List[ToolCall]


# ============================================================================
# Calendar Agent
# ============================================================================

class CalendarAgent:
    """Azure OpenAI-powered calendar scheduling agent."""
    
    # Default system prompt for calendar operations
    DEFAULT_SYSTEM_PROMPT = """You are a helpful calendar scheduling assistant with access to powerful tools for managing calendars, emails, and users.

IMPORTANT: You MUST use the available tools to complete user requests. Do NOT try to complete tasks manually or describe what you would do - actually use the tools. Please do not respond asking for more details, use default values if you need to fill in missing parameters for a tool call.

Available capabilities:
- Send emails using sendMail
- Search for messages using searchMessages
- Create, list, and manage calendar events
- Find and list users in the organization

When a user asks you to perform an action (like "send an email" or "schedule a meeting"), you should:
1. Use the appropriate tool to perform the action
2. Confirm what you did after the tool executes successfully

Always prefer using tools over describing what should be done."""
    
    def __init__(self, mcp_server_url: Optional[str] = None, system_prompt: Optional[str] = None):
        # Initialize Azure OpenAI client
        # Prefer API key (for Docker), fallback to Entra ID (for local dev)
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://oai-exp.openai.azure.com/")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if api_key:
            logger.info("Using API key authentication for Azure OpenAI")
            self.client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
        else:
            logger.info("Using Entra ID (DefaultAzureCredential) authentication for Azure OpenAI")
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version
            )
        
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
        
        # Store custom system prompt or use default
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # MCP server configuration
        self.mcp_server_url = mcp_server_url or os.getenv("MCP_SERVER_URL")
        self.mcp_session: Optional[ClientSession] = None
        self.mcp_connected = False
        self._mcp_read = None
        self._mcp_write = None
        self._mcp_context = None
        self._get_session_id = None
        
        # Tools will be populated from MCP server
        self.tools = []
        
        # Correlation headers for MCP tool calls
        self.correlation_headers: Dict[str, str] = {}
    
    async def connect_mcp(self):
        """Connect to MCP server over HTTP with streamable transport and keep connection alive."""
        await self.connect_mcp_with_headers()
    
    async def connect_mcp_with_headers(self, headers: Optional[Dict[str, str]] = None):
        """Connect to MCP server with optional correlation headers."""
        if not self.mcp_server_url:
            logger.warning("No MCP server URL configured - agent will not have any tools available")
            logger.warning("Set MCP_SERVER_URL environment variable or pass mcp_server_url parameter")
            return
        
        try:
            logger.info(f"Attempting to connect to MCP server at {self.mcp_server_url}")
            
            # Use streamable HTTP client (newer transport, replaces deprecated SSE)
            logger.info(f"Connecting with streamable HTTP transport to: {self.mcp_server_url}")
            
            # Pass correlation headers to MCP server
            mcp_headers = headers or {}
            if self.correlation_headers:
                mcp_headers.update(self.correlation_headers)
                logger.info(f"Using correlation headers for MCP connection: {list(mcp_headers.keys())}")
            
            # Create streamable HTTP client connection
            # Returns: (read_stream, write_stream, get_session_id_callback)
            self._mcp_context = streamablehttp_client(self.mcp_server_url, headers=mcp_headers)
            read, write, get_session_id = await self._mcp_context.__aenter__()
            self._mcp_read = read
            self._mcp_write = write
            self._get_session_id = get_session_id
            
            logger.info(f"MCP HTTP connection established, session ID callback available")
            
            logger.info("MCP HTTP connection established, creating session...")
            
            # Create session - keep it alive by not using context manager
            self.mcp_session = ClientSession(read, write)
            await self.mcp_session.__aenter__()
            
            logger.info("MCP client session created, initializing...")
            await self.mcp_session.initialize()
            logger.info("MCP session initialized successfully")
            
            self.mcp_connected = True
            
            # List available tools from MCP server
            logger.info("Requesting tool list from MCP server...")
            tools_result = await self.mcp_session.list_tools()
            logger.info(f"Connected to MCP server with {len(tools_result.tools)} tools: {[t.name for t in tools_result.tools]}")
            
            # Update tool definitions from MCP server
            self._update_tools_from_mcp(tools_result.tools)
            logger.info(f"Tool definitions updated from MCP server")
                    
        except ConnectionError as e:
            logger.error(f"Connection error to MCP server at {self.mcp_server_url}: {e}", exc_info=True)
            logger.error("Agent will not have any tools available")
            self.mcp_connected = False
        except TimeoutError as e:
            logger.error(f"Timeout connecting to MCP server at {self.mcp_server_url}: {e}", exc_info=True)
            logger.error("Agent will not have any tools available")
            self.mcp_connected = False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {type(e).__name__}: {e}", exc_info=True)
            logger.error("Agent will not have any tools available")
            self.mcp_connected = False
    
    async def disconnect_mcp(self):
        """Disconnect from MCP server and cleanup resources."""
        if self.mcp_session:
            try:
                await self.mcp_session.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing MCP session: {e}")
        
        if self._mcp_context:
            try:
                await self._mcp_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing MCP context: {e}")
        
        self.mcp_connected = False
        logger.info("Disconnected from MCP server")
    
    def _update_tools_from_mcp(self, mcp_tools):
        """Update OpenAI function definitions from MCP tool schemas."""
        # Convert MCP tool definitions to OpenAI function calling format
        self.tools = []
        for tool in mcp_tools:
            self.tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                }
            })
    
    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Process user request and return response with tool uses."""
        
        logger.info(f"Processing request for test case: {request.test_case_id}")
        
        # Initialize conversation with user message
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": request.input
            }
        ]
        
        tool_calls = []
        max_iterations = 10
        
        for iteration in range(max_iterations):
            # Call Azure OpenAI
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # If no tool calls, we're done
            if not message.tool_calls:
                final_response = message.content or "Task completed."
                logger.info(f"🏁 Agent finished without tool calls. Response: {final_response[:100]}...")
                break
            
            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"🔧 TOOL CALL: {function_name}")
                logger.info(f"   Arguments: {json.dumps(function_args, indent=2)}")
                
                # Execute the tool
                result = await self._execute_tool(function_name, function_args)
                logger.info(f"✅ TOOL RESULT: {function_name} completed")
                
                # Record tool call with response
                tool_calls.append(ToolCall(
                    name=function_name,
                    arguments=[
                        ToolArgument(name=k, value=v)
                        for k, v in function_args.items()
                    ],
                    response=result  # Capture the MCP tool response
                ))
                
                # Add tool response to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            # Max iterations reached
            final_response = "Task completed after maximum iterations."
            logger.warning(f"⚠️  Max iterations ({max_iterations}) reached")
        
        # Summary logging
        logger.info(f"📊 EXECUTION SUMMARY:")
        logger.info(f"   Total tool calls: {len(tool_calls)}")
        if tool_calls:
            for i, tc in enumerate(tool_calls, 1):
                logger.info(f"   {i}. {tc.name}")
        else:
            logger.warning(f"   ⚠️  NO TOOLS WERE CALLED!")
        
        return InvokeResponse(
            response=final_response,
            tool_calls=tool_calls
        )
    
    async def _execute_tool(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool function via MCP or fallback to mocks."""
        
        # Try MCP server first if connected
        if self.mcp_connected and self.mcp_session:
            try:
                logger.info(f"→ Executing MCP tool: {function_name}")
                result = await self.mcp_session.call_tool(function_name, arguments)
                logger.info(f"← MCP tool {function_name} responded successfully")
                
                # Parse MCP response
                if result.content:
                    content = result.content[0]
                    logger.info(f"MCP response content type: {type(content)}, hasattr text: {hasattr(content, 'text')}")
                    if hasattr(content, 'text'):
                        parsed = json.loads(content.text)
                        logger.info(f"Successfully parsed MCP response: {parsed}")
                        return parsed
                    else:
                        result_str = str(content)
                        logger.info(f"MCP content as string: {result_str}")
                        return {"result": result_str}
                logger.warning(f"MCP tool {function_name} returned no content")
                return {"status": "success"}
                
            except json.JSONDecodeError as e:
                logger.error(f"MCP tool {function_name} returned invalid JSON: {e}, content: {content.text if hasattr(content, 'text') else content}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"MCP tool call failed for {function_name}: {type(e).__name__}: {e}", exc_info=True)
                raise
        
        # No MCP connection - return error
        error_msg = f"Tool {function_name} not available - MCP server not connected"
        logger.error(error_msg)
        return {"error": error_msg, "mcp_connected": False}


# ============================================================================
# Simple Agent Prompts
# ============================================================================
# These prompts are optimized for the simple evaluation datasets and have been
# tested to pass all test cases.
# ============================================================================

SIMPLE_EMAIL_PROMPT = """You are a helpful email assistant. Your job is to send emails exactly as requested by the user.

IMPORTANT: You MUST use the sendMail tool to send emails. Do not just describe what you would do - actually call the tool.

## INSTRUCTIONS:

When the user asks you to send an email, use the sendMail tool with these parameters:

1. **toRecipients**: Array of email addresses to send to
   - Extract the email address(es) from the user's request
   - Format: ["email@example.com"]

2. **subject**: A clear, relevant subject line
   - Create a subject that summarizes the main topic of the email
   - Should be concise but informative

3. **body**: The full email body content
   - Include ALL the information the user asked you to include
   - Be thorough - don't leave out any details from the request
   - Use professional formatting with clear sections if there are multiple points
   - Include greetings and sign-offs as appropriate

## AFTER SENDING THE EMAIL:

After successfully calling the sendMail tool, you MUST provide a confirmation that:
1. States the email was sent successfully
2. Lists the recipient(s)
3. Summarizes ALL the key points that were included in the email

Example confirmation:
"Email sent successfully to coaches@example.com with the following information:
- Meeting date changed to Saturday May 18th
- Championship confirmed for Memorial Day weekend
- Referee assignments completed by Mike
- Volunteer needs communicated
- Photo schedule at 10 AM
- Roster deadline reminder for Thursday"

## CRITICAL RULES:
1. ALWAYS call the sendMail tool - do not just describe the email
2. Include ALL details from the user's request in the email body
3. Use the exact email address(es) provided by the user
4. Create a professional, clear email
5. ALWAYS provide a detailed confirmation listing what was included

Now send the email as requested."""


SIMPLE_MEETING_PROMPT = """You are a helpful calendar and meeting assistant. Your job is to manage calendar events and send emails as requested by the user.

IMPORTANT: You MUST use the available tools to complete user requests. Do not just describe what you would do - actually call the tools.

## AVAILABLE TOOLS:

1. **mcp_CalendarTools_graph_listEvents** - List calendar events
   - Use this to check calendar availability, find conflicts, or summarize upcoming meetings
   - Parameters:
     - startDateTime: Start of time range (ISO 8601 format, e.g., "2024-01-15T00:00:00Z")
     - endDateTime: End of time range (ISO 8601 format)
   
2. **mcp_CalendarTools_graph_createEvent** - Create calendar events
   - Use this to schedule new meetings
   - Parameters:
     - subject: Meeting title/subject
     - start: Start datetime (ISO 8601 format with timezone)
     - end: End datetime (ISO 8601 format with timezone)
     - body: Meeting description/agenda (optional)
     - attendees: Array of attendee email addresses (optional)
     - isOnlineMeeting: Set to true for Teams meetings (optional)

3. **sendMail** - Send emails
   - Use this to notify people about meetings, send confirmations, etc.
   - Parameters:
     - toRecipients: Array of email addresses
     - subject: Email subject line
     - body: Email body content

## WORKFLOW PATTERNS:

### Creating a Meeting:
1. Call mcp_CalendarTools_graph_createEvent with proper parameters
2. Confirm with details of what was scheduled

### Checking Calendar / Finding Conflicts:
1. Call mcp_CalendarTools_graph_listEvents for the relevant time range
2. Review events to find conflicts or availability
3. Report findings to the user

### Rescheduling a Meeting:
1. Call mcp_CalendarTools_graph_listEvents to check current schedule
2. Find an available time slot
3. Call mcp_CalendarTools_graph_createEvent to create at new time
4. Call sendMail to notify affected attendees about the change

### Summarizing Schedule:
1. Call mcp_CalendarTools_graph_listEvents for the requested time period
2. Provide a clear summary of all meetings with times and subjects

## AFTER COMPLETING THE TASK:

After successfully completing the request, you MUST provide a detailed confirmation that:
1. States what action was completed
2. Lists ALL relevant details (times, subjects, attendees, etc.)
3. Summarizes what was included in any emails sent

Example confirmations:

For meeting creation:
"I've scheduled the Daily Standup meeting for tomorrow at 9:00 AM - 9:30 AM. The meeting will be held online via Microsoft Teams."

For calendar summary:
"Here's your schedule for this week:
- Monday 10:00 AM: Team Sync (1 hour)
- Wednesday 2:00 PM: Client Call (30 minutes)
- Friday 3:00 PM: Weekly Review (1 hour)"

For rescheduling:
"I found a conflict at 3 PM tomorrow with your existing Weekly Review meeting. I've rescheduled your meeting with John Doe to 4:00 PM - 5:00 PM and sent him an email explaining the change with:
- Original time: 3 PM
- New time: 4 PM
- Reason: Calendar conflict"

## CRITICAL RULES:
1. ALWAYS call the appropriate tools - do not just describe actions
2. Use proper ISO 8601 datetime formats with timezone
3. Include ALL details from the user's request
4. ALWAYS provide detailed confirmation listing what was done
5. When sending emails about meetings, include all relevant meeting details

Now complete the calendar/meeting request."""


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Calendar Agent", version="2.0.0")

mcp_server_url = os.getenv("MCP_SERVER_URL")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "agents": {
            "calendar": "/agents/calendar/invoke",
            "simple": {
                "email": "/agents/simple/email/invoke",
                "meeting": "/agents/simple/meeting/invoke",
            }
        }
    }


async def _invoke_agent_with_prompt(
    system_prompt: str,
    agent_name: str,
    request: InvokeRequest,
    http_request: Request
) -> InvokeResponse:
    """Common handler for invoking any agent with a specific system prompt."""
    try:
        # Extract correlation headers
        correlation_headers = {}
        correlation_id = http_request.headers.get('x-correlationid')
        test_case_id = http_request.headers.get('x-testcaseid')

        if correlation_id:
            correlation_headers['x-correlationid'] = correlation_id
        if test_case_id:
            correlation_headers['x-testcaseid'] = test_case_id

        logger.info(f"Processing {agent_name} request with correlation headers: {correlation_headers}")

        # Create a fresh agent instance with custom system prompt
        request_agent = CalendarAgent(mcp_server_url, system_prompt=system_prompt)

        # Connect with correlation headers if we have any
        if correlation_headers:
            await request_agent.connect_mcp_with_headers(correlation_headers)
        else:
            await request_agent.connect_mcp()

        try:
            # Process the request with the dedicated agent instance
            response = await request_agent.invoke(request)
            return response
        finally:
            # Always cleanup the per-request agent
            await request_agent.disconnect_mcp()

    except Exception as e:
        logger.error(f"Error invoking {agent_name} agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Simple Agent Endpoints
# ============================================================================

@app.post("/agents/simple/email/invoke", response_model=InvokeResponse)
async def invoke_simple_email_agent(request: InvokeRequest, http_request: Request):
    """Invoke simple email agent for basic email sending tasks."""
    return await _invoke_agent_with_prompt(
        SIMPLE_EMAIL_PROMPT,
        "simple-email",
        request,
        http_request
    )


@app.get("/agents/simple/email")
async def simple_email_info():
    """Get information about the simple email agent."""
    return {
        "name": "simple-email",
        "description": "Simple email collaboration agent for basic email tasks",
        "dataset": "Email Collaboration Dataset",
        "deployment": deployment_name,
    }


@app.post("/agents/simple/meeting/invoke", response_model=InvokeResponse)
async def invoke_simple_meeting_agent(request: InvokeRequest, http_request: Request):
    """Invoke simple meeting agent for basic calendar and meeting tasks."""
    return await _invoke_agent_with_prompt(
        SIMPLE_MEETING_PROMPT,
        "simple-meeting",
        request,
        http_request
    )


@app.get("/agents/simple/meeting")
async def simple_meeting_info():
    """Get information about the simple meeting agent."""
    return {
        "name": "simple-meeting",
        "description": "Simple meeting scheduler agent for calendar and meeting tasks",
        "dataset": "Meeting Scheduler Dataset",
        "deployment": deployment_name,
    }


# ============================================================================
# Calendar Agent Endpoint (Original)
# ============================================================================

@app.post("/agents/calendar/invoke", response_model=InvokeResponse)
async def invoke_agent(request: InvokeRequest, http_request: Request):
    """Invoke the calendar agent with a user request."""
    try:
        # Extract correlation headers
        correlation_headers = {}
        correlation_id = http_request.headers.get('x-correlationid')
        test_case_id = http_request.headers.get('x-testcaseid')
        
        if correlation_id:
            correlation_headers['x-correlationid'] = correlation_id
        if test_case_id:
            correlation_headers['x-testcaseid'] = test_case_id
            
        logger.info(f"Processing request with correlation headers: {correlation_headers}")
        
        # Create a fresh agent instance with its own MCP connection for this request
        request_agent = CalendarAgent(mcp_server_url)
        
        # Connect with correlation headers if we have any
        if correlation_headers:
            await request_agent.connect_mcp_with_headers(correlation_headers)
        else:
            await request_agent.connect_mcp()
        
        try:
            # Process the request with the dedicated agent instance
            response = await request_agent.invoke(request)
            return response
        finally:
            # Always cleanup the per-request agent
            await request_agent.disconnect_mcp()
            
    except Exception as e:
        logger.error(f"Error invoking agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/calendar")
async def agent_info():
    """Get information about the calendar agent."""
    return {
        "name": "calendar",
        "description": "Azure OpenAI-powered calendar scheduling agent",
        "deployment": deployment_name,
        "mcp_server_url": mcp_server_url,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Calendar Agent server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
