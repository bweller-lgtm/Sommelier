#!/usr/bin/env python3
"""Launch the Taster MCP server for Claude Desktop integration."""
import asyncio
import os
import sys
import time
from pathlib import Path

# Fix Windows encoding — subprocess pipes default to cp1252 which breaks on Unicode.
# PYTHONIOENCODING in the env (set in claude_desktop_config.json) handles startup,
# but we also reconfigure here for any streams that got created before the env took effect.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

# Ensure working directory is the project root so config.yaml, profiles/, etc. are found
os.chdir(Path(__file__).parent)

t0 = time.time()
print("[taster] Importing modules...", file=sys.stderr, flush=True)
from taster.mcp.server import create_mcp_server
print(f"[taster] Imports done in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)


def main():
    server = create_mcp_server()
    init_options = server.create_initialization_options()

    async def run():
        from mcp.server.stdio import stdio_server
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)

    asyncio.run(run())


if __name__ == "__main__":
    main()
