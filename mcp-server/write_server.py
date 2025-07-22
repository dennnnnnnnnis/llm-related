import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WriteServer")
USER_AGENT = "write-app/1.0"

OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

@mcp.tool()
async def write_file(content: str) -> str:
    """
    Write content to local directory as a new file, return the file name.
    :param content: things that will be written into the new file
    :return: file path
    """
    # file name with timestamp
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    filename = f"note_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        # write into the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Write to the file successfully."
    except Exception as e:
        return f"Fail to write to the file: {e}"
    
if __name__ == "__main__":
    mcp.run(transport="stdio")