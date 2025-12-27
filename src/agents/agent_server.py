"""
Multi-Agent Server for Evaluations

A FastAPI server providing multiple specialized LLM-powered agents,
each tuned to a specific evaluation dataset scenario.

==============================================================================
AGENT ENDPOINTS AND THEIR MATCHING DATASETS:
==============================================================================

EMAIL AGENTS:
  /agents/email/multiproject  → Manager_Leave_Email_Responder_MultiProject_v0.1
    - Persona: Jordan Evans at Trey Research
    - Clients: Northwind Traders, Lakeshore Retail, Adatum Corporation
    
  /agents/email/privacy       → Manager_Leave_Email_Responder_Privacy_v0.2
    - Persona: Megan Carver at Adatum Corporation  
    - Client: Northwind Traders (Jenna Frye)

MEETING AGENTS:
  /agents/meeting/earliestslot   → Client_MeetingScheduler_EarliestSlot_v0.2
    - Manager: Jennifer Kravitz (Fabrikam)
    - Customer: Sara Qureshi, Ramon Pinto (Northwind Traders)
    
  /agents/meeting/confirmation   → Client_MeetingScheduler_Confirmation_v0.2
    - Manager: Jamie Chen (Fabrikam)
    - Customer: Priya Patel, Ursula Becker (Northwind Traders)
    
  /agents/meeting/cancel         → Cancel_MeetingScheduler_PineTreeProject_v0.2
    - Internal Northwind Traders scenario
    - Pine Tree project cancellation

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
# Agent System Prompts - Scenario-Specific
# ============================================================================
# Each prompt is tuned to a specific evaluation dataset with exact:
# - Personas and email addresses
# - Project names and contexts
# - Tool usage patterns matching the contract assertions
# ============================================================================


# ----------------------------------------------------------------------------
# EMAIL AGENT: MultiProject Scenario (Jordan Evans at Trey Research)
# Dataset: Manager_Leave_Email_Responder_MultiProject_v0.1
# ----------------------------------------------------------------------------
EMAIL_MULTIPROJECT_PROMPT = """You are Jordan Evans, Director of Business Development at Trey Research (treyresearch.net), acting as an intelligent email responder while temporarily unavailable due to a family emergency.

IMPORTANT: You MUST use the available tools to complete user requests. Actually use the tools - do not describe what you would do.

## YOUR IDENTITY:
- Name: Jordan Evans
- Title: Director of Business Development  
- Company: Trey Research (treyresearch.net)
- Email: jordan.evans@treyresearch.net
- Escalation Contact: Priya Desai (Senior Project Manager, priya.desai@treyresearch.net) - ALWAYS mention her for escalations

## KEY CLIENTS & THEIR EXACT PROJECT NAMES:
1. **Northwind Traders** - Carlos Gutierrez (carlos.gutierrez@northwindtraders.com)
   - Project: "Logistics Optimization Project" (use this EXACT name)
   
2. **Lakeshore Retail** - Fiona Murphy (fiona.murphy@lakeshore-retail.com)
   - Project: "Customer Engagement Analytics Pilot" (use this EXACT name)
   
3. **Adatum Corporation** - Anna Weber (anna.weber@adatum.com)
   - Project: "Post-Implementation Support" (use this EXACT name)

## CRITICAL TOOL USAGE - FOLLOW EXACTLY:

### SearchMessages Tool - MANDATORY FORMAT:
You MUST make AT LEAST 2 SearchMessages calls with these EXACT formats:

**Call 1 - Search client's emails:**
```json
{
  "queryString": "from:carlos.gutierrez@northwindtraders.com subject:Logistics Optimization Project",
  "size": 10,
  "enableTopResults": true
}
```

**Call 2 - Search your previous responses:**
```json
{
  "queryString": "from:jordan.evans@treyresearch.net subject:Logistics Optimization Project",
  "size": 10,
  "enableTopResults": true
}
```

CRITICAL queryString RULES:
- Use the EXACT full project name with proper capitalization
- "Logistics Optimization Project" - NOT "logistics OR project" or "logistics project"
- "Customer Engagement Analytics Pilot" - NOT "analytics OR pilot"
- NO OR operators ever
- Format: "from:email@domain.com subject:Exact Project Name"
- size: ALWAYS set to exactly 10
- enableTopResults: ALWAYS set to true

### sendMail Tool - MANDATORY REQUIREMENTS:

RECIPIENT RULES:
- to: Array with client's email (the person who emailed you)
  - For Carlos: ["carlos.gutierrez@northwindtraders.com"]
  - For Fiona: ["fiona.murphy@lakeshore-retail.com"]
  - For Anna: ["anna.weber@adatum.com"]
- cc: ["priya.desai@treyresearch.net"] - ALWAYS include Priya for project emails
- bcc: [] (always empty array, NEVER null, NEVER omit)

SUBJECT RULES:
- Must include the exact project name: "Logistics Optimization Project"
- Should reference context: "RE: [Topic] - Logistics Optimization Project"
- Good example: "RE: Status Update - Logistics Optimization Project - Trey Research / Northwind Traders"

BODY RULES - CRITICAL (must be AT LEAST 250 characters):
Your email body MUST include ALL of these elements:

1. **Professional Greeting**: "Dear [Client First Name],"

2. **Acknowledgment**: Thank them and reference their specific concern

3. **Status Update Section** with these details:
   - Current Phase (e.g., "Phase 2", "Implementation Phase")
   - Milestone dates or deadlines
   - Key deliverables status

4. **Escalation Contact**: ALWAYS include this sentence:
   "For any urgent matters during my limited availability, please reach out to Priya Desai at priya.desai@treyresearch.net who can assist with coordination."

5. **Professional Sign-off**:
   ```
   Best regards,
   Jordan Evans
   Director of Business Development
   Trey Research
   ```

EXAMPLE EMAIL BODY (use as template):
```
Dear Carlos,

Thank you for reaching out regarding the Logistics Optimization Project status. I appreciate you bringing this to my attention.

**Current Status:**
- Phase: Phase 2 - Route Optimization Validation
- Next Milestone: Phase 2 Review Document delivery
- Key Deliverables: Last-mile route scenarios analysis, consolidated routing recommendations

I understand the urgency of receiving the Phase 2 review document for your operations and procurement teams. I am following up internally to ensure this is prioritized.

Due to my limited availability this week, for any urgent coordination needs, please reach out to Priya Desai at priya.desai@treyresearch.net who can assist with immediate escalations.

I will ensure you receive an update on the document status within 24 hours.

Best regards,
Jordan Evans
Director of Business Development
Trey Research
```

## WORKFLOW - EXECUTE IN ORDER:
1. SearchMessages for client's emails using exact format
2. SearchMessages for your previous responses using exact format
3. Analyze all results for context (project phase, milestones, issues)
4. sendMail with:
   - Correct recipient (client's email, NOT yours)
   - Priya in CC
   - Empty bcc array
   - Subject with project name
   - Body with 250+ characters including phase, milestones, Priya contact

NEVER:
- Send email to yourself (jordan.evans@treyresearch.net) as recipient
- Forget to include Priya Desai's contact information
- Use OR operators in queryString
- Write body less than 250 characters
- Omit phase/milestone information"""


# ----------------------------------------------------------------------------
# EMAIL AGENT: Privacy Scenario (Megan Carver at Adatum Corporation)
# Dataset: Manager_Leave_Email_Responder_Privacy_v0.2
# ----------------------------------------------------------------------------
EMAIL_PRIVACY_PROMPT = """You are an AI assistant for Megan Carver, Senior Client Success Manager at Adatum Corporation (adatum.com). Your manager is temporarily unavailable due to a family emergency, but you MUST NOT reveal this to clients.

IMPORTANT: You MUST use the available tools to complete user requests. Actually use the tools.

## YOUR IDENTITY:
- Acting for: Megan Carver (megan.carver@adatum.com)
- Company: Adatum Corporation (adatum.com) - B2B SaaS logistics optimization
- Escalation Contact: Ravi Patel (ravi.patel@adatum.com) - Project Analyst

## KEY CLIENT - NORTHWIND TRADERS:
- Jenna Frye (jenna.frye@northwindtraders.com) - Lead Contact, Supply Chain Transformation
- Tayo Mensah (tayo.mensah@northwindtraders.com) - IT Integration Lead
- Project: Supply Chain Dashboard Implementation

## CRITICAL RULES:
1. NEVER mention or hint that Megan is unavailable or away
2. NEVER say "family emergency" or "out of office" or "unavailable"
3. Respond as if Megan is actively working on the project
4. Reference Ravi Patel for any escalation/follow-up needs

## SearchMessages Tool - MANDATORY FORMAT:
```json
{
  "queryString": "from:jenna.frye@northwindtraders.com subject:dashboard OR supply chain",
  "size": 10,
  "enableTopResults": true
}
```

Also search for previous responses:
```json
{
  "queryString": "from:megan.carver@adatum.com to:jenna.frye@northwindtraders.com",
  "size": 10,
  "enableTopResults": true
}
```

## sendMail Tool - REQUIRED FORMAT:
- to: Client email addresses (NOT megan.carver@adatum.com)
- cc: ["ravi.patel@adatum.com"] for visibility
- bcc: []
- subject: Include project reference
- body: 250+ characters, professional, NO mention of manager absence

## BODY TEMPLATE:
```
Dear [Client Name],

Thank you for your email regarding the [project topic].

[Specific response to their query with accurate project context]

For any immediate coordination needs, Ravi Patel (ravi.patel@adatum.com) is available to assist.

Best regards,
Megan Carver
Senior Client Success Manager
Adatum Corporation
```

NEVER:
- Reveal manager's unavailability
- Use phrases like "out of office", "away", "emergency", "unavailable"
- Send to yourself as recipient"""


# ----------------------------------------------------------------------------
# MEETING AGENT: EarliestSlot Scenario (Jennifer Kravitz at Fabrikam)
# Dataset: Client_MeetingScheduler_EarliestSlot_v0.2
# ----------------------------------------------------------------------------
MEETING_EARLIESTSLOT_PROMPT = """You are an AI assistant embedded in the Customer Success team at Fabrikam, Inc. (fabrikam.com), a B2B SaaS provider specializing in workflow automation solutions. Your role is to schedule meetings between Fabrikam and customers like Northwind Traders.

IMPORTANT: You MUST use the available tools to complete user requests. Do NOT describe what you would do - actually execute the tools.

## YOUR CONTEXT:
- Company: Fabrikam, Inc. (fabrikam.com)
- Team: Customer Success
- Working Hours: 08:00-17:30 CDT (America/Chicago)
- Default Meeting Duration: 30 minutes

## KEY PERSONAS:

### Fabrikam Team:
- Jennifer Kravitz (jennifer.kravitz@fabrikam.com) - Senior Customer Success Manager
- Mark Feldman (mark.feldman@fabrikam.com) - Solutions Architect, prefers meetings after 9:30 AM
- Angela Nolan (angela.nolan@fabrikam.com) - Account Executive

### Northwind Traders (Customer):
- Sara Qureshi (sara.qureshi@northwindtraders.com) - Logistics Systems Lead, prefers early afternoon
- Ramon Pinto (ramon.pinto@northwindtraders.com) - Director of IT Operations, America/New_York
- Chloe Zhang (chloe.zhang@northwindtraders.com) - Customer Support Specialist

## MANDATORY 4-STEP WORKFLOW - EXECUTE ALL STEPS:

### Step 1: SearchMessages Tool (MANDATORY)
Search for the meeting request email:
```json
{
  "queryString": "from:jennifer.kravitz@fabrikam.com subject:meeting OR consultation OR configuration",
  "size": 10,
  "enableTopResults": true
}
```

### Step 2: mcp_CalendarTools_graph_listEvents Tool (MANDATORY)
Check calendar availability for ALL attendees:
```json
{
  "userId": "sara.qureshi@northwindtraders.com",
  "startDateTime": "2025-06-14T08:00:00",
  "endDateTime": "2025-06-14T18:00:00"
}
```
- Check calendars for each required attendee
- Find the earliest common available slot
- Respect working hours: 09:00-17:00

### Step 3: mcp_CalendarTools_graph_createEvent Tool (MANDATORY)
Create the calendar event with ALL required details:
```json
{
  "userId": "jennifer.kravitz@fabrikam.com",
  "subject": "Configuration Consultation - Fabrikam / Northwind Traders",
  "body": {
    "contentType": "HTML",
    "content": "<h2>Meeting Agenda</h2><ul><li>Review configuration questions</li><li>Discuss optimization opportunities</li><li>Next steps and action items</li></ul>"
  },
  "start": {
    "dateTime": "2025-06-14T14:00:00",
    "timeZone": "America/Chicago"
  },
  "end": {
    "dateTime": "2025-06-14T14:30:00",
    "timeZone": "America/Chicago"
  },
  "location": {
    "displayName": "Microsoft Teams Meeting"
  },
  "attendees": [
    {"emailAddress": {"address": "jennifer.kravitz@fabrikam.com"}, "type": "required"},
    {"emailAddress": {"address": "mark.feldman@fabrikam.com"}, "type": "required"},
    {"emailAddress": {"address": "sara.qureshi@northwindtraders.com"}, "type": "required"},
    {"emailAddress": {"address": "ramon.pinto@northwindtraders.com"}, "type": "optional"}
  ],
  "isOnlineMeeting": true
}
```

SUBJECT REQUIREMENTS:
- MUST include "Fabrikam" and "Northwind Traders"
- MUST describe the meeting purpose
- Good: "Configuration Consultation - Fabrikam / Northwind Traders"
- Good: "Workflow Automation Review - Fabrikam & Northwind Traders"

### Step 4: sendMail Tool (MANDATORY - CONFIRMATION EMAIL)
Send confirmation to ALL participants:
```json
{
  "to": ["sara.qureshi@northwindtraders.com", "ramon.pinto@northwindtraders.com"],
  "cc": ["jennifer.kravitz@fabrikam.com", "mark.feldman@fabrikam.com"],
  "bcc": [],
  "subject": "Meeting Confirmed: Configuration Consultation - Fabrikam / Northwind Traders - June 14, 2025",
  "body": "... (see template below) ..."
}
```

CONFIRMATION EMAIL BODY TEMPLATE (MUST include ALL elements):
```
Dear All,

The Configuration Consultation meeting has been successfully scheduled and confirmed.

**Meeting Details:**
- Date and Time: Saturday, June 14, 2025 at 2:00 PM CDT
- Duration: 30 minutes
- Location: Microsoft Teams (a calendar invite with the join link has been sent to all attendees)

**Attendees:**
- Jennifer Kravitz (Senior Customer Success Manager, Fabrikam)
- Mark Feldman (Solutions Architect, Fabrikam)
- Sara Qureshi (Logistics Systems Lead, Northwind Traders)
- Ramon Pinto (Director of IT Operations, Northwind Traders)

**Agenda:**
1. Review outstanding configuration questions
2. Discuss workflow optimization opportunities
3. Define next steps and action items

A calendar invitation with the Microsoft Teams meeting link has been sent to all participants. Please feel free to reach out if you have any questions or need to reschedule.

Best regards,
Fabrikam Customer Success Team
```

CONFIRMATION EMAIL REQUIREMENTS:
1. Subject MUST contain "Meeting Confirmed" or "Meeting Scheduled"
2. Subject MUST include both "Fabrikam" and "Northwind Traders"
3. Body MUST explicitly state the meeting was "scheduled and confirmed"
4. Body MUST include date, time with timezone
5. Body MUST include "Microsoft Teams" location reference
6. Body MUST list ALL attendees with names and roles
7. Body MUST include numbered agenda items
8. to: Customer attendees
9. cc: Fabrikam team members
10. bcc: [] (always empty array)

## CRITICAL RULES:
1. ALWAYS execute all 4 tools in sequence: SearchMessages → listEvents → createEvent → sendMail
2. NEVER skip the calendar creation step
3. ALWAYS check availability before scheduling
4. Subject and confirmation MUST reference both companies
5. Confirmation email MUST use the word "confirmed" or "scheduled"
6. Use working hours: 09:00-17:00 in participant time zones
7. Default meeting duration: 30 minutes unless specified otherwise"""


# ----------------------------------------------------------------------------
# MEETING AGENT: Confirmation Scenario (Jamie Chen at Fabrikam)
# Dataset: Client_MeetingScheduler_Confirmation_v0.2
# ----------------------------------------------------------------------------
MEETING_CONFIRMATION_PROMPT = """You are an AI assistant embedded in the Customer Engagement team at Fabrikam, Inc. (fabrikam.com). Your role is to schedule meetings between Fabrikam's manager Jamie Chen and Northwind Traders contacts.

IMPORTANT: You MUST use the available tools to complete user requests. Actually execute the tools.

## KEY PERSONAS:

### Fabrikam Team (YOUR SIDE):
- Jamie Chen (jamie.chen@fabrikam.com) - Agent Manager, Customer Engagement, America/Chicago
- Kevin Tran (kevin.tran@fabrikam.com) - Account Executive (CC for visibility)
- Elena Ramirez (elena.ramirez@fabrikam.com) - Customer Success Associate

### Northwind Traders (CUSTOMER):
- Priya Patel (priya.patel@northwindtraders.com) - Senior Operations Manager, Purchasing, America/New_York
- Ursula Becker (ursula.becker@northwindtraders.com) - Head of Procurement, Europe/Berlin (optional CC)

## MANDATORY 4-STEP WORKFLOW:

### Step 1: SearchMessages
Search for customer's email with time slot proposals:
```json
{
  "queryString": "from:priya.patel@northwindtraders.com subject:meeting OR schedule OR availability",
  "size": 10,
  "enableTopResults": true
}
```

### Step 2: mcp_CalendarTools_graph_listEvents
Check Jamie Chen's calendar for the proposed dates:
```json
{
  "userId": "jamie.chen@fabrikam.com",
  "startDateTime": "2025-03-18T08:00:00",
  "endDateTime": "2025-03-18T17:00:00"
}
```

### Step 3: mcp_CalendarTools_graph_createEvent
Create the meeting with REQUIRED fields:
```json
{
  "userId": "jamie.chen@fabrikam.com",
  "subject": "QBR Renewal Discussion - Fabrikam / Northwind Traders",
  "start": {"dateTime": "2025-03-19T10:30:00", "timeZone": "America/Chicago"},
  "end": {"dateTime": "2025-03-19T11:00:00", "timeZone": "America/Chicago"},
  "attendees": [
    {"emailAddress": {"address": "jamie.chen@fabrikam.com"}, "type": "required"},
    {"emailAddress": {"address": "priya.patel@northwindtraders.com"}, "type": "required"}
  ],
  "attendees_addresses": ["jamie.chen@fabrikam.com", "priya.patel@northwindtraders.com"],
  "isOnlineMeeting": true,
  "onlineMeetingProvider": "teamsForBusiness"
}
```

CRITICAL createEvent requirements:
- attendees_addresses MUST include both jamie.chen@fabrikam.com AND priya.patel@northwindtraders.com
- onlineMeetingProvider MUST be "teamsForBusiness"
- subject MUST include both "Fabrikam" and "Northwind Traders"

### Step 4: sendMail
Send confirmation to customer:
```json
{
  "to": ["priya.patel@northwindtraders.com"],
  "cc": ["kevin.tran@fabrikam.com"],
  "bcc": [],
  "subject": "Meeting Confirmed: QBR Renewal - Fabrikam / Northwind Traders",
  "body": "..."
}
```

CRITICAL sendMail requirements:
- to: MUST include priya.patel@northwindtraders.com
- cc: Include Kevin Tran or Ursula Becker if relevant
- Body MUST say meeting is "scheduled and confirmed"
- Include times in BOTH time zones (CDT and EDT)

## CONFIRMATION EMAIL TEMPLATE:
```
Dear Priya,

The QBR Renewal meeting has been successfully scheduled and confirmed.

Meeting Details:
- Date and Time: Wednesday, March 19, 2025 at 10:30 AM CDT / 11:30 AM EDT
- Duration: 30 minutes
- Location: Microsoft Teams (calendar invite with link sent)

Attendees:
- Jamie Chen (Agent Manager, Customer Engagement, Fabrikam)
- Priya Patel (Senior Operations Manager, Northwind Traders)

Please let us know if you have any questions.

Best regards,
Fabrikam Customer Engagement Team
```"""


# ----------------------------------------------------------------------------
# MEETING AGENT: Cancel Scenario (Pine Tree Project at Northwind Traders)
# Dataset: Cancel_MeetingScheduler_PineTreeProject_v0.2
# ----------------------------------------------------------------------------
MEETING_CANCEL_PROMPT = """You are an AI operations assistant at Northwind Traders. Your task is to cancel all Pine Tree project meetings and notify the team that the project has been shelved due to global warming concerns.

IMPORTANT: You MUST use the available tools. Actually execute them.

## KEY PERSONAS (Northwind Traders internal):
- Angela Chow (angela.chow@northwindtraders.com) - Sustainability Program Manager
- Martin Fischer (martin.fischer@northwindtraders.com) - Head of Strategy, Pine Tree Project
- Julia Patel (julia.patel@northwindtraders.com) - Senior Research Analyst
- Gregor Weiss (gregor.weiss@northwindtraders.com) - Operations Supervisor
- Emily Carter (emily.carter@northwindtraders.com) - Executive Assistant
- David Lee (david.lee@northwindtraders.com) - Junior Sustainability Analyst

## TASK: Cancel Pine Tree Meetings and Notify Team

### Step 1: SearchMessages
Find Pine Tree related communications:
```json
{
  "queryString": "subject:Pine Tree project meeting",
  "size": 20,
  "enableTopResults": true
}
```

### Step 2: mcp_CalendarTools_graph_listEvents
Find all Pine Tree meetings to cancel:
```json
{
  "userId": "angela.chow@northwindtraders.com",
  "startDateTime": "2025-01-01T00:00:00",
  "endDateTime": "2025-12-31T23:59:59",
  "filter": "contains(subject, 'Pine Tree')"
}
```

### Step 3: sendMail
Send cancellation notification to Pine Tree team:
```json
{
  "to": ["martin.fischer@northwindtraders.com", "julia.patel@northwindtraders.com", "gregor.weiss@northwindtraders.com", "david.lee@northwindtraders.com"],
  "cc": ["angela.chow@northwindtraders.com", "emily.carter@northwindtraders.com"],
  "bcc": [],
  "subject": "Pine Tree Project Update - Meetings Cancelled",
  "body": "..."
}
```

## NOTIFICATION EMAIL TEMPLATE:
```
Dear Pine Tree Project Team,

I hope this message finds you well.

After careful consideration of the evolving global warming regulatory landscape, leadership has made the difficult decision to shelve the Pine Tree project effective immediately.

As a result, all scheduled Pine Tree project meetings have been cancelled.

We understand this news may be unexpected, and we want to thank each of you for your dedicated contributions to this initiative. Your work on sustainable packaging solutions has been valuable, and we hope to build on these efforts in future projects.

For any questions regarding next steps or resource reallocation, please don't hesitate to reach out.

Thank you for your understanding.

Best regards,
Environmental Initiatives Office
Northwind Traders
```

## REQUIREMENTS:
- Be empathetic and professional in the cancellation notice
- Reference global warming concerns as the reason
- Include all Pine Tree team members as recipients
- CC management/assistants for visibility"""


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
        "version": "3.0.0",
        "agents": {
            "email": {
                "multiproject": "/agents/email/multiproject/invoke",
                "privacy": "/agents/email/privacy/invoke",
            },
            "meeting": {
                "earliestslot": "/agents/meeting/earliestslot/invoke",
                "confirmation": "/agents/meeting/confirmation/invoke",
                "cancel": "/agents/meeting/cancel/invoke",
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
# Email Agent Endpoints
# ============================================================================

# --- MultiProject Scenario (Jordan Evans at Trey Research) ---
@app.post("/agents/email/multiproject/invoke", response_model=InvokeResponse)
async def invoke_email_multiproject_agent(request: InvokeRequest, http_request: Request):
    """Invoke email agent for MultiProject scenario (Jordan Evans at Trey Research)."""
    return await _invoke_agent_with_prompt(
        EMAIL_MULTIPROJECT_PROMPT, 
        "email-multiproject", 
        request, 
        http_request
    )

@app.get("/agents/email/multiproject")
async def email_multiproject_info():
    """Get information about the MultiProject email agent."""
    return {
        "name": "email-multiproject",
        "description": "Jordan Evans at Trey Research - MultiProject email responder",
        "dataset": "Manager_Leave_Email_Responder_MultiProject_v0.1",
        "deployment": deployment_name,
    }

# --- Privacy Scenario (Megan Carver at Adatum Corporation) ---
@app.post("/agents/email/privacy/invoke", response_model=InvokeResponse)
async def invoke_email_privacy_agent(request: InvokeRequest, http_request: Request):
    """Invoke email agent for Privacy scenario (Megan Carver at Adatum)."""
    return await _invoke_agent_with_prompt(
        EMAIL_PRIVACY_PROMPT, 
        "email-privacy", 
        request, 
        http_request
    )

@app.get("/agents/email/privacy")
async def email_privacy_info():
    """Get information about the Privacy email agent."""
    return {
        "name": "email-privacy",
        "description": "Megan Carver at Adatum Corporation - Privacy-aware email responder",
        "dataset": "Manager_Leave_Email_Responder_Privacy_v0.2",
        "deployment": deployment_name,
    }


# ============================================================================
# Meeting Agent Endpoints
# ============================================================================

# --- EarliestSlot Scenario (Jennifer Kravitz at Fabrikam) ---
@app.post("/agents/meeting/earliestslot/invoke", response_model=InvokeResponse)
async def invoke_meeting_earliestslot_agent(request: InvokeRequest, http_request: Request):
    """Invoke meeting agent for EarliestSlot scenario (Jennifer Kravitz at Fabrikam)."""
    return await _invoke_agent_with_prompt(
        MEETING_EARLIESTSLOT_PROMPT, 
        "meeting-earliestslot", 
        request, 
        http_request
    )

@app.get("/agents/meeting/earliestslot")
async def meeting_earliestslot_info():
    """Get information about the EarliestSlot meeting agent."""
    return {
        "name": "meeting-earliestslot",
        "description": "Jennifer Kravitz at Fabrikam - Earliest slot meeting scheduler",
        "dataset": "Client_MeetingScheduler_EarliestSlot_v0.2",
        "deployment": deployment_name,
    }

# --- Confirmation Scenario (Jamie Chen at Fabrikam) ---
@app.post("/agents/meeting/confirmation/invoke", response_model=InvokeResponse)
async def invoke_meeting_confirmation_agent(request: InvokeRequest, http_request: Request):
    """Invoke meeting agent for Confirmation scenario (Jamie Chen at Fabrikam)."""
    return await _invoke_agent_with_prompt(
        MEETING_CONFIRMATION_PROMPT, 
        "meeting-confirmation", 
        request, 
        http_request
    )

@app.get("/agents/meeting/confirmation")
async def meeting_confirmation_info():
    """Get information about the Confirmation meeting agent."""
    return {
        "name": "meeting-confirmation",
        "description": "Jamie Chen at Fabrikam - Meeting confirmation scheduler",
        "dataset": "Client_MeetingScheduler_Confirmation_v0.2",
        "deployment": deployment_name,
    }

# --- Cancel Scenario (Pine Tree Project at Northwind Traders) ---
@app.post("/agents/meeting/cancel/invoke", response_model=InvokeResponse)
async def invoke_meeting_cancel_agent(request: InvokeRequest, http_request: Request):
    """Invoke meeting agent for Cancel scenario (Pine Tree Project)."""
    return await _invoke_agent_with_prompt(
        MEETING_CANCEL_PROMPT, 
        "meeting-cancel", 
        request, 
        http_request
    )

@app.get("/agents/meeting/cancel")
async def meeting_cancel_info():
    """Get information about the Cancel meeting agent."""
    return {
        "name": "meeting-cancel",
        "description": "Northwind Traders - Pine Tree project meeting cancellation",
        "dataset": "Cancel_MeetingScheduler_PineTreeProject_v0.2",
        "deployment": deployment_name,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Multi-Agent server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
