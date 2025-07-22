import os
import json
import httpx
from typing import Any
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("DemoServer")

OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/3.0/onecall?"
API_KEY = ""
USER_AGENT = "weather-app/1.0"

# Asynchronous programming allows your code to handle multiple tasks 
# concurrently without blocking. When you have operations that take time 
# (like network requests, file I/O, or database queries), async lets your 
# program continue doing other work while waiting for those operations to complete.
async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    Get weather info through OpenWeather API
    :param city: city name
    :return: weather data in dict, or dict with error information
    """
    params = {
        "lat": 40,
        "lon": 100,
        "appid": API_KEY,
        "units": "metric",
        "lang": "en"
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(OPENWEATHER_API_BASE, params=params, 
                                        headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

def format_weather(data: dict[str, Any] | str) -> str:
    """
    Formate the weather data to readable form.
    :param data: weather data, could be dict or Json
    :return: formatted weather data string
    """

    # if input is string, convert to dict
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"Wrong weather data: {e}"
        
    if "error" in data:
        return f"{data["error"]}"
    
    # city = data.get("name", "Unknown")
    # country = data.get("sys", {}).get("country", "Unknown")
    place = data.get("timezone", "Unknown")
    temp = data.get("current", {}).get("temp", "N/A")

    return (
        f"{place}\n"
        f"Current Temperature: {temp}\n"
    )

@mcp.tool()
async def query_weather(city: str) -> str:
    """
    Input city name, return the queried answer (weather)
    :param city: city name
    :return: formated weather data
    """
    data = await fetch_weather(city)
    return format_weather(data)

if __name__ == "__main__":
    mcp.run(transport="stdio")