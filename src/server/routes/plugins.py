from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

PLUGIN_DIRS = ["plugins", "src/plugins/user_plugins"]


def register_plugin_routes(app: "FastAPI") -> None:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse

    from src.plugins import PluginManager

    manager = PluginManager(plugin_dirs=PLUGIN_DIRS)
    discovered = manager.discover()

    router = APIRouter(prefix="/api/plugins", tags=["plugins"])

    @router.get("/")
    async def list_plugins() -> list[dict]:
        return manager.list_plugins()

    @router.post("/discover")
    async def discover_plugins() -> dict:
        found = manager.discover()
        return {"status": "ok", "discovered": found, "total": len(manager.plugins)}

    @router.post("/load")
    async def load_plugin(data: dict[str, Any]) -> dict:
        path = data.get("path", "")
        if not path:
            raise HTTPException(status_code=400, detail="path required")
        plugin = manager.load_single(path)
        if not plugin:
            raise HTTPException(status_code=400, detail="Failed to load plugin")
        return {"status": "ok", "name": plugin.name, "version": plugin.version}

    @router.post("/unload")
    async def unload_plugin(data: dict[str, Any]) -> dict:
        name = data.get("name", "")
        if not name:
            raise HTTPException(status_code=400, detail="name required")
        if manager.unload(name):
            return {"status": "ok"}
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")

    @router.get("/routes")
    async def plugin_routes() -> list[dict]:
        return manager.get_all_routes()

    @router.post("/hooks/chat")
    async def run_chat_hooks(data: dict[str, Any]) -> dict:
        prompt = data.get("prompt", "")
        response = data.get("response", "")
        modified_prompt, modified_response = manager.run_hook("on_chat", prompt, response)
        return {"prompt": modified_prompt, "response": modified_response}

    app.include_router(router)
