#!/usr/bin/env python3
"""Launch the Taste Cloner MCP server for Claude Desktop integration."""
import asyncio
from src.mcp.server import create_mcp_server


def main():
    server = create_mcp_server()

    async def run():
        from mcp.server.stdio import stdio_server
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

    asyncio.run(run())


if __name__ == "__main__":
    main()
