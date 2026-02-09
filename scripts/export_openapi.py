"""Export OpenAPI spec from FastAPI app without starting the server.

Usage:
    uv run scripts/export_openapi.py > web/frontend-v2/openapi.json

The spec is used by openapi-ts to generate TypeScript types.
"""

import sys

import orjson

from web.server import create_app

app = create_app()
spec = app.openapi()
sys.stdout.buffer.write(orjson.dumps(spec, option=orjson.OPT_INDENT_2))
sys.stdout.buffer.write(b"\n")
