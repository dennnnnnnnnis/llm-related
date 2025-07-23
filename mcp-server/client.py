import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from prompts import *

# LLM API key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# set up memory saver so llm can store histories
checkpointer = InMemorySaver()

# dialog config, if multiple user chat, each user respect to a new thread id
config = {
    "configurable": {
        "thread_id": "1"
    }
}

# mcp servers config
def load_servers(file_path: str = "servers_config.json") -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f).get("mcpServers", {})
    
async def run_chat_loop() -> None:
    servers_config = load_servers()

    # convert all the external mcp tools to langchain tools and save to mcp_client
    mcp_client = MultiServerMCPClient(servers_config)

    # langchain tool list
    tools = await mcp_client.get_tools()
        
    logging.info(f"Load {len(tools)} MCP tools: {[t.name for t in tools]}")

    # llm define
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # $0.60 / 1M input tokens
        api_key=API_KEY
    )

    # langgraph agent
    agent = create_react_agent(model=llm, tools=tools, prompt=AGENT_SYSTEM_PROMPT, checkpointer=checkpointer)

    # CLI Chat
    print("\n MCP Agent initiated, enter 'quit' to exit")
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )
            print(f"\nAssistant: {result["messages"][-1].content}")
        except Exception as e:
            print(f"\nError: {e}")
    
    await mcp_client.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run_chat_loop())