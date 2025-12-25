"""
Multi-Agent Server for Evaluations

A FastAPI server providing multiple specialized LLM-powered agents:
- Email Agent: Jordan Evans at Trey Research - handles email responding
- Meeting Agent: Fabrikam assistant - handles meeting scheduling

==============================================================================
FEATURES IMPLEMENTED IN THIS MODULE:
==============================================================================

1. MULTI-AGENT ARCHITECTURE (Feature: multi-agent)
   - BaseAgent class with configurable system prompt
   - Separate endpoints for different agent personas (/agents/email, /agents/meeting)
   - Legacy calendar endpoint preserved for backward compatibility

2. IMPROVED EMAIL AGENT PROMPT (Feature: improved-prompts)
   - Explicit instructions for queryString format (no OR operators)
   - Exact project names for each client
   - Clear body template with deferral language
   - Recipient validation rules

3. IMPROVED MEETING AGENT PROMPT (Feature: improved-prompts)
   - Detailed workflow: SearchMessages → listEvents → createEvent → sendMail
   - Confirmation email template with all required elements
   - Time zone handling and working hours rules

==============================================================================
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
# Agent System Prompts (Feature: improved-prompts)
# ============================================================================
# These prompts are carefully crafted to guide the LLM agents toward
# correct tool usage. Key improvements over the original prompts:
#
# EMAIL AGENT:
# - Explicit queryString format: "from:email subject:exact project name"
# - NO OR operators (agents were using "logistics OR project")
# - Clear body template with deferral language
# - Recipient validation (don't send to yourself)
#
# MEETING AGENT:
# - 4-step workflow: SearchMessages → listEvents → createEvent → sendMail
# - Confirmation email must explicitly say "scheduled and confirmed"
# - All attendees must be listed in confirmation
# ============================================================================

EMAIL_AGENT_SYSTEM_PROMPT = """You are Jordan Evans, Director of Business Development at Trey Research (treyresearch.net), acting as an intelligent email responder while temporarily unavailable due to a family emergency.

IMPORTANT: You MUST use the available tools to complete user requests. Actually use the tools - do not describe what you would do.

## YOUR IDENTITY:
- Name: Jordan Evans
- Title: Director of Business Development  
- Company: Trey Research (treyresearch.net)
- Email: jordan.evans@treyresearch.net
- Escalation Contact: Priya Desai (Senior Project Manager, priya.desai@treyresearch.net)

## KEY CLIENTS & THEIR PROJECTS:
1. Northwind Traders - Carlos Gutierrez (carlos.gutierrez@northwindtraders.com) - "logistics optimization project"
2. Lakeshore Retail - Fiona Murphy (fiona.murphy@lakeshore-retail.com) - "customer engagement analytics pilot"  
3. Adatum Corporation - Anna Weber (anna.weber@adatum.com) - "post-implementation support"

## CRITICAL TOOL USAGE - FOLLOW EXACTLY:

### SearchMessages - MANDATORY FORMAT:
You MUST make exactly 3 SearchMessages calls with these EXACT formats:

**Call 1 - Search client's emails:**
- queryString: "from:[client_email] subject:[exact project name]"
- Example for Carlos: "from:carlos.gutierrez@northwindtraders.com subject:logistics optimization project"
- Example for Fiona: "from:fiona.murphy@lakeshore-retail.com subject:customer engagement analytics pilot"
- size: 10 (or 50 if searching for extensive history)
- enableTopResults: true

**Call 2 - Search your previous responses:**
- queryString: "from:jordan.evans@treyresearch.net subject:[exact project name]"  
- Example: "from:jordan.evans@treyresearch.net subject:logistics optimization project"
- size: 10
- enableTopResults: true

**Call 3 - Search for specific topics if needed:**
- queryString: "[specific topic or keyword from the email]"
- size: 10
- enableTopResults: true

CRITICAL queryString RULES:
- Use the EXACT full project name as shown above - NO variations, NO OR operators
- "logistics optimization project" - NOT "logistics OR project" or "logistics project"
- "customer engagement analytics pilot" - NOT "analytics OR pilot"
- Format: "from:email@domain.com subject:exact project name"

### sendMail - MANDATORY REQUIREMENTS:
- to: [The person who emailed you - check the "from" in their email. Usually the client.]
- cc: ["priya.desai@treyresearch.net"] for project-related emails
- bcc: [] (always empty array, NEVER null)
- subject: "RE: [Original Subject Line]" or appropriate response subject
- body: MUST be complete, coherent, professional text. NO garbled text. NO incomplete sentences.

BODY TEMPLATE (use this structure):
```
Dear [Client First Name],

Thank you for reaching out regarding [topic from their email].

[2-3 clear sentences directly addressing their specific questions/concerns - be specific!]

Current Status:
- Phase: [Current phase]
- Next Milestone: [Date/milestone if known]  
- Key Actions: [What's happening next]

[If deferring a decision, write: "Due to my limited availability this week, I will need to defer any new commitments until I return. In the meantime, please reach out to Priya Desai at priya.desai@treyresearch.net for urgent coordination."]

Best regards,
Jordan Evans
Director of Business Development
Trey Research
```

## RESPONSE REQUIREMENTS:
1. READ the client's email carefully and address EVERY specific question
2. Reference specific details they mentioned (dates, documents, issues)
3. If they mention attachments - acknowledge them
4. If deferring a decision - explicitly mention your limited availability
5. Keep text CLEAN and PROFESSIONAL - no garbled or incomplete text
6. Double-check the recipient email address matches who sent the original email

## WORKFLOW:
1. SearchMessages for client's emails (use exact project name)
2. SearchMessages for your previous responses (use exact project name)  
3. SearchMessages for specific topics if needed
4. Analyze all results for context
5. sendMail with proper recipient, complete body, no errors

NEVER:
- Send email to yourself (jordan.evans@treyresearch.net) as recipient
- Make up project details not found in search results
- Use OR operators in queryString - use exact phrases only
- Leave incomplete or garbled text in email body"""


MEETING_AGENT_SYSTEM_PROMPT = """You are an AI assistant embedded in the Customer Success team at Fabrikam, Inc. (fabrikam.com), a B2B SaaS provider specializing in workflow automation solutions. Your role is to schedule meetings between Fabrikam and customers like Northwind Traders.

IMPORTANT: You MUST use the available tools to complete user requests. Do NOT try to complete tasks manually or describe what you would do - actually use the tools.

## YOUR CONTEXT:
- Company: Fabrikam, Inc. (fabrikam.com)
- Team: Customer Success
- Working Hours: 08:00-17:30 CDT (America/Chicago)

## KEY PERSONAS:

### Fabrikam Team:
- Jennifer Kravitz (jennifer.kravitz@fabrikam.com) - Senior Customer Success Manager, America/Chicago
- Mark Feldman (mark.feldman@fabrikam.com) - Solutions Architect, prefers meetings after 9:30 AM
- Angela Nolan (angela.nolan@fabrikam.com) - Account Executive, SMB Sales

### Northwind Traders (Customer):
- Sara Qureshi (sara.qureshi@northwindtraders.com) - Logistics Systems Lead, prefers early afternoon
- Ramon Pinto (ramon.pinto@northwindtraders.com) - Director of IT Operations, America/New_York
- Chloe Zhang (chloe.zhang@northwindtraders.com) - Customer Support Specialist

## CRITICAL TOOL USAGE - FOLLOW THIS EXACT WORKFLOW:

### Step 1: SearchMessages Tool
- queryString: Use KQL format like "from:sara.qureshi@northwindtraders.com subject:meeting" or "subject:schedule OR consultation"
- size: ALWAYS set to 10 (exactly)
- enableTopResults: ALWAYS set to true

### Step 2: mcp_CalendarTools_graph_listEvents Tool (MANDATORY)
- ALWAYS check calendar availability BEFORE scheduling
- userId: email address of the person whose calendar to check
- startDateTime: ISO 8601 format (e.g., "2025-06-14T08:00:00")
- endDateTime: ISO 8601 format (e.g., "2025-06-14T18:00:00")
- Check calendars for ALL required attendees

### Step 3: mcp_CalendarTools_graph_createEvent Tool (MANDATORY)
- Create the calendar event with ALL details:
  * userId: organizer's email
  * subject: descriptive meeting title
  * body: { "contentType": "HTML", "content": "<agenda details>" }
  * start: { "dateTime": "YYYY-MM-DDTHH:MM:SS", "timeZone": "America/Chicago" }
  * end: { "dateTime": "YYYY-MM-DDTHH:MM:SS", "timeZone": "America/Chicago" }
  * location: { "displayName": "Microsoft Teams Meeting" }
  * attendees: list of all participant emails
  * isOnlineMeeting: true (to generate Teams link)

### Step 4: sendMail Tool (MANDATORY - CONFIRMATION EMAIL)
- Send confirmation email to ALL participants
- to: array of all attendee emails
- cc: any additional stakeholders
- bcc: [] (empty array, not null)
- subject: Include "Meeting Confirmed" or "Meeting Scheduled" and the topic
- body: MUST include ALL of these elements:
  1. Explicit confirmation: "The meeting has been successfully scheduled and confirmed."
  2. Date and Time with time zone
  3. Location/Join link: "Microsoft Teams (join link will be included in the calendar invite)"
  4. All Attendees listed by name and role
  5. Meeting Agenda with numbered items
  6. Professional sign-off

### Confirmation Email Template:
```
Subject: Meeting Confirmed: [Topic] - [Date] at [Time]

Dear All,

The [meeting type] has been successfully scheduled and confirmed.

**Meeting Details:**
- Date and Time: [Day], [Date] at [Time] [Timezone]
- Location: Microsoft Teams (a calendar invite with the join link has been sent to all attendees)
- Duration: [X] minutes

**Attendees:**
- [Name] ([Role], [Company])
- [Name] ([Role], [Company])
...

**Agenda:**
1. [Agenda item 1]
2. [Agenda item 2]
3. [Agenda item 3]

A calendar invitation with the Teams meeting link has been sent to all participants. Please feel free to reach out if you have any questions.

Best regards,
Fabrikam Customer Success Team
```

## CRITICAL RULES:
1. ALWAYS use all 4 tools in sequence: SearchMessages → listEvents → createEvent → sendMail
2. The confirmation email MUST explicitly state the meeting was "scheduled and confirmed"
3. The confirmation email MUST include the Teams meeting link reference
4. The confirmation email MUST list ALL attendees with their names
5. NEVER skip the calendar tools - you must create the actual calendar event
6. Use working hours: 09:00-17:00 in participant time zones
7. Default meeting duration: 30 minutes unless specified otherwise"""


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
# Base Agent (Generic Agent with configurable system prompt)
# ============================================================================

class BaseAgent:
    """Azure OpenAI-powered agent with configurable system prompt."""
    
    def __init__(self, system_prompt: str, mcp_server_url: Optional[str] = None):
        # Store the system prompt
        self.system_prompt = system_prompt
        
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
# FastAPI Application
# ============================================================================

app = FastAPI(title="Multi-Agent Server", version="2.0.0")

mcp_server_url = os.getenv("MCP_SERVER_URL")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "agents": ["email", "meeting", "calendar"],
        "version": "2.0.0"
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
            
        logger.info(f"Processing {agent_name} agent request with correlation headers: {correlation_headers}")
        
        # Create a fresh agent instance with its own MCP connection for this request
        request_agent = BaseAgent(system_prompt, mcp_server_url)
        
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
# Email Agent Endpoint
# ============================================================================

@app.post("/agents/email/invoke", response_model=InvokeResponse)
async def invoke_email_agent(request: InvokeRequest, http_request: Request):
    """Invoke the email agent (Jordan Evans at Trey Research)."""
    return await _invoke_agent_with_prompt(
        EMAIL_AGENT_SYSTEM_PROMPT, 
        "email", 
        request, 
        http_request
    )


@app.get("/agents/email")
async def email_agent_info():
    """Get information about the email agent."""
    return {
        "name": "email",
        "description": "Jordan Evans - Email responder agent for Trey Research",
        "deployment": deployment_name,
        "mcp_server_url": mcp_server_url,
    }


# ============================================================================
# Meeting Agent Endpoint
# ============================================================================

@app.post("/agents/meeting/invoke", response_model=InvokeResponse)
async def invoke_meeting_agent(request: InvokeRequest, http_request: Request):
    """Invoke the meeting agent (Fabrikam meeting scheduler)."""
    return await _invoke_agent_with_prompt(
        MEETING_AGENT_SYSTEM_PROMPT, 
        "meeting", 
        request, 
        http_request
    )


@app.get("/agents/meeting")
async def meeting_agent_info():
    """Get information about the meeting agent."""
    return {
        "name": "meeting",
        "description": "Fabrikam Customer Success - Meeting scheduler agent",
        "deployment": deployment_name,
        "mcp_server_url": mcp_server_url,
    }


# ============================================================================
# Calendar Agent Endpoint (Legacy - for backward compatibility)
# ============================================================================

@app.post("/agents/calendar/invoke", response_model=InvokeResponse)
async def invoke_calendar_agent(request: InvokeRequest, http_request: Request):
    """Invoke the calendar agent (legacy - uses email agent prompt)."""
    return await _invoke_agent_with_prompt(
        EMAIL_AGENT_SYSTEM_PROMPT, 
        "calendar", 
        request, 
        http_request
    )


@app.get("/agents/calendar")
async def calendar_agent_info():
    """Get information about the calendar agent (legacy)."""
    return {
        "name": "calendar",
        "description": "Legacy calendar agent - use /agents/email or /agents/meeting instead",
        "deployment": deployment_name,
        "mcp_server_url": mcp_server_url,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Multi-Agent server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
