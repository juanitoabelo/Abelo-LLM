from __future__ import annotations

from src.plugins.manager import PluginBase


class ExamplePlugin(PluginBase):
    name = "example"
    version = "0.1.0"
    description = "Example plugin that logs all chats"

    def on_chat(self, prompt: str, response: str) -> tuple[str, str]:
        print(f"[ExamplePlugin] Chat: {len(prompt)} chars in, {len(response)} chars out")
        return prompt, response

    def on_tool_call(self, tool_name: str, args: dict) -> dict:
        print(f"[ExamplePlugin] Tool called: {tool_name}")
        return args

    def get_routes(self) -> list[dict]:
        return [
            {"method": "GET", "path": "/api/plugins/example/hello", "description": "Example health check"},
        ]
