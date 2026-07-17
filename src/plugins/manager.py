from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any


class PluginBase:
    name: str = "unnamed_plugin"
    version: str = "0.1.0"
    description: str = ""

    def on_load(self) -> None:
        pass

    def on_unload(self) -> None:
        pass

    def on_chat(self, prompt: str, response: str) -> tuple[str, str]:
        return prompt, response

    def on_tool_call(self, tool_name: str, args: dict) -> dict:
        return args

    def on_tool_result(self, tool_name: str, result: Any) -> Any:
        return result

    def get_routes(self) -> list[dict]:
        return []


class PluginManager:
    def __init__(self, plugin_dirs: list[str | Path] | None = None) -> None:
        self.plugin_dirs = [Path(d) for d in (plugin_dirs or [])]
        self.plugins: dict[str, PluginBase] = {}
        self._hooks: dict[str, list[str]] = {
            "on_chat": [],
            "on_tool_call": [],
            "on_tool_result": [],
        }

    def discover(self) -> list[str]:
        found: list[str] = []
        for d in self.plugin_dirs:
            if not d.exists():
                continue
            for f in d.glob("*.py"):
                if f.stem.startswith("_"):
                    continue
                plugin = self._load_plugin(f)
                if plugin:
                    self.plugins[plugin.name] = plugin
                    found.append(plugin.name)
                    for hook in self._hooks:
                        if hasattr(plugin, hook) and callable(getattr(plugin, hook)):
                            self._hooks[hook].append(plugin.name)
                    plugin.on_load()
        return found

    def load_single(self, file_path: str | Path) -> Optional["PluginBase"]:
        plugin = self._load_plugin(Path(file_path))
        if plugin:
            self.plugins[plugin.name] = plugin
            plugin.on_load()
        return plugin

    def unload(self, name: str) -> bool:
        if name in self.plugins:
            self.plugins[name].on_unload()
            del self.plugins[name]
            for hook_list in self._hooks.values():
                if name in hook_list:
                    hook_list.remove(name)
            return True
        return False

    def run_hook(self, hook: str, *args, **kwargs) -> Any:
        result = None
        for plugin_name in self._hooks.get(hook, []):
            plugin = self.plugins.get(plugin_name)
            if plugin:
                method = getattr(plugin, hook)
                result = method(*args, **kwargs)
                if result is not None:
                    args = (result,) if args else result
        return result if result else (args[0] if args else None)

    def get_all_routes(self) -> list[dict]:
        routes: list[dict] = []
        for plugin in self.plugins.values():
            routes.extend(plugin.get_routes())
        return routes

    def list_plugins(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "hooks": [h for h in self._hooks if p.name in self._hooks[h]],
            }
            for p in self.plugins.values()
        ]

    def _load_plugin(self, path: Path) -> Optional["PluginBase"]:
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if not spec or not spec.loader:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for _, obj in inspect.getmembers(mod, lambda x: inspect.isclass(x) and issubclass(x, PluginBase) and x is not PluginBase):
                return obj()
        except Exception:
            pass
        return None
