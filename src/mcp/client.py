from __future__ import annotations

import json
from typing import Any, Optional
from urllib.request import Request, urlopen


class MCPClient:
    def __init__(self, server_url: str, api_key: Optional[str] = None) -> None:
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def list_tools(self) -> list[dict]:
        try:
            req = Request(
                f"{self.server_url}/tools/list",
                headers=self._headers(),
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get("tools", [])
        except Exception:
            return []

    def call_tool(self, name: str, arguments: dict) -> dict:
        payload = json.dumps({"name": name, "arguments": arguments}).encode()
        req = Request(
            f"{self.server_url}/tools/call",
            data=payload,
            headers=self._headers(),
        )
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}

    def list_resources(self) -> list[dict]:
        try:
            req = Request(
                f"{self.server_url}/resources/list",
                headers=self._headers(),
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get("resources", [])
        except Exception:
            return []

    def read_resource(self, uri: str) -> Optional[str]:
        payload = json.dumps({"uri": uri}).encode()
        req = Request(
            f"{self.server_url}/resources/read",
            data=payload,
            headers=self._headers(),
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data.get("content")
        except Exception:
            return None


class MCPToolAdapter:
    def __init__(self, client: MCPClient) -> None:
        self.client = client

    def get_tool_definitions(self) -> list[dict]:
        tools = self.client.list_tools()
        specs: list[dict] = []
        for tool in tools:
            specs.append({
                "type": "function",
                "function": {
                    "name": f"mcp_{tool.get('name', 'unknown')}",
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                },
            })
        return specs
