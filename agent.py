#!/usr/bin/env python3
"""
Travel Planner Agent

An interactive travel planning assistant powered by Claude Opus 4.6.
Connects to the travel MCP server and uses its tools to help users
plan trips — flights, hotels, itineraries, costs, weather, and more.

Usage:
    python agent.py
    python agent.py --prompt "Plan a 5-day trip to Tokyo in April"
"""

import os
import asyncio
import sys
import argparse
import json
from pathlib import Path

import anthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_PATH = Path(__file__).parent / "mcp_server.py"
MODEL       = "claude-opus-4-6"
MAX_TOKENS  = 8192

SYSTEM_PROMPT = """You are an expert travel planner with decades of experience helping travellers worldwide.

You help users plan amazing trips by:
- Searching for flights and finding the best deals
- Discovering hotels that match their style and budget
- Building day-by-day itineraries tailored to their interests
- Providing accurate weather forecasts so they pack appropriately
- Estimating trip costs with a full budget breakdown
- Checking visa and passport requirements before they travel
- Sharing currency tips and daily budget guidance
- Recommending the best attractions, restaurants, and experiences

Always be enthusiastic, friendly, and thorough. When a user asks for trip help, proactively
use multiple tools to give a comprehensive answer — for example, when planning a trip, check
flights, hotels, weather, and attractions in one response.

Ask clarifying questions when you need key details (dates, budget level, interests, origin city).
Format your responses clearly with sections and bullet points for easy reading."""

WELCOME = """
╔══════════════════════════════════════════════════════╗
║           ✈  Travel Planner  —  Powered by Claude   ║
╚══════════════════════════════════════════════════════╝

I can help you with:
  • Searching flights & comparing prices
  • Finding hotels that fit your budget
  • Building day-by-day itineraries
  • Weather forecasts for your travel dates
  • Visa & entry requirements
  • Budget estimates & currency tips
  • Top attractions & hidden gems

Type 'quit' or press Ctrl+C to exit.
──────────────────────────────────────────────────────
"""

EXAMPLE_PROMPTS = [
    "Plan a 7-day trip to Paris leaving from New York in June",
    "What's the weather like in Bali in August?",
    "Find flights from London to Tokyo for 2 people next month",
    "What visa do I need to visit Japan as a US citizen?",
    "How much does a moderate budget trip to Barcelona cost for 5 days?",
]

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TravelAgent:
    """Interactive travel planner that connects to the MCP server and Claude."""

    def __init__(self, api_key: str | None = None) -> None:
        self.client    = anthropic.AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.messages: list[dict] = []
        self.tools:    list[dict] = []

    async def connect(self, session: ClientSession) -> None:
        """Discover tools from the MCP server and convert to Anthropic format."""
        tools_result = await session.list_tools()
        self.tools = [
            {
                "name":         t.name,
                "description":  t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in tools_result.tools
        ]
        print(f"Connected to travel MCP server — {len(self.tools)} tools available\n")

    async def chat(self, session: ClientSession, user_input: str) -> str:
        """Send a user message and return the final assistant response."""
        self.messages.append({"role": "user", "content": user_input})

        # Inner tool-use loop
        while True:
            # Stream text tokens for real-time output, then collect the full message
            full_text = ""
            first_token = True

            async with self.client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.messages,
                thinking={"type": "adaptive"},
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            if first_token:
                                print("\nAssistant: ", end="", flush=True)
                                first_token = False
                            print(event.delta.text, end="", flush=True)
                            full_text += event.delta.text

                response = await stream.get_final_message()

            # Build assistant content list (text + tool_use blocks)
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type":  "tool_use",
                        "id":    block.id,
                        "name":  block.name,
                        "input": block.input,
                    })

            self.messages.append({"role": "assistant", "content": assistant_content})

            # If no tool calls, we're done
            if response.stop_reason != "tool_use":
                if first_token and full_text == "":
                    # Extract text from content blocks if streaming missed it
                    for block in response.content:
                        if block.type == "text":
                            full_text = block.text
                            print(f"\nAssistant: {full_text}", end="", flush=True)
                print()  # trailing newline
                return full_text

            # Execute tool calls
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            print()  # newline after any partial text

            tool_results = []
            for tool_block in tool_use_blocks:
                print(f"\n  [Tool] {tool_block.name}({json.dumps(tool_block.input, separators=(',', ':'))})")

                try:
                    mcp_result = await session.call_tool(tool_block.name, tool_block.input)
                    result_text = "\n".join(
                        c.text for c in mcp_result.content if hasattr(c, "text")
                    ) or "{}"
                    is_error = False
                except Exception as exc:
                    result_text = json.dumps({"error": str(exc)})
                    is_error = True

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_block.id,
                    "content":     result_text,
                    "is_error":    is_error,
                })

            self.messages.append({"role": "user", "content": tool_results})
            # Continue loop — Claude will now respond with the tool results

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages = []
        print("\n[Conversation reset]\n")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def interactive_mode(agent: TravelAgent, session: ClientSession) -> None:
    """Run an interactive REPL conversation."""
    print(WELCOME)
    print("Example prompts:")
    for i, p in enumerate(EXAMPLE_PROMPTS, 1):
        print(f"  {i}. {p}")
    print()

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Have a wonderful trip! ✈️\n")
            break

        user_input = user_input.strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("\nGoodbye! Have a wonderful trip! ✈️\n")
            break

        if user_input.lower() in ("reset", "clear", "new"):
            agent.reset()
            continue

        if user_input.lower() in ("help", "?"):
            print("\nCommands:")
            print("  reset / clear / new  — start a new conversation")
            print("  quit / exit / bye    — exit the agent")
            print("  help / ?             — show this message\n")
            continue

        try:
            await agent.chat(session, user_input)
        except anthropic.RateLimitError:
            print("\n[Rate limit reached — please wait a moment and try again]\n")
        except anthropic.APIError as exc:
            print(f"\n[API error {exc.status_code}: {exc.message}]\n")


async def single_prompt_mode(agent: TravelAgent, session: ClientSession, prompt: str) -> None:
    """Run a single prompt and print the result."""
    await agent.chat(session, prompt)
    print()


async def run(prompt: str | None = None) -> None:
    """Connect to the MCP server and start the agent."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = TravelAgent()
            await agent.connect(session)

            if prompt:
                await single_prompt_mode(agent, session, prompt)
            else:
                await interactive_mode(agent, session)


def main() -> None:
    parser = argparse.ArgumentParser(description="Travel Planner Agent powered by Claude Opus 4.6")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Run a single prompt instead of interactive mode")
    args = parser.parse_args()

    asyncio.run(run(prompt=args.prompt))


if __name__ == "__main__":
    main()
