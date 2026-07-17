from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI


def register_structured_routes(app: "FastAPI") -> None:
    from fastapi import APIRouter, HTTPException

    router = APIRouter(prefix="/api/structured", tags=["structured"])

    @router.post("/generate")
    async def structured_generate(data: dict[str, Any]) -> dict:
        prompt = data.get("prompt", "")
        schema = data.get("schema", {})
        model = data.get("model", "llama3.2:1b")
        temperature = data.get("temperature", 0.1)

        if not prompt or not schema:
            raise HTTPException(status_code=400, detail="prompt and schema required")

        schema_str = _format_schema(schema)
        sys_prompt = f"""You are a structured data extractor. Output ONLY valid JSON matching this schema:

{schema_str}

Rules:
- Return ONLY the JSON object, no other text
- Follow the schema exactly
- Use null for missing values
- All string values must use double quotes"""

        full_prompt = f"{sys_prompt}\n\nInput: {prompt}\n\nOutput:"
        result = _call_ollama(full_prompt, model, temperature)
        parsed = _parse_json_response(result)
        if parsed is None:
            raise HTTPException(status_code=500, detail="Failed to parse structured output")

        return {"status": "ok", "data": parsed, "raw": result}

    @router.post("/extract")
    async def extract_entities(data: dict[str, Any]) -> dict:
        text = data.get("text", "")
        entity_types = data.get("entity_types", ["person", "organization", "location", "date"])
        model = data.get("model", "llama3.2:1b")

        if not text:
            raise HTTPException(status_code=400, detail="text required")

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": entity_types},
                            "context": {"type": "string"},
                        },
                    },
                }
            },
        }

        return await structured_generate({
            "prompt": f"Extract {', '.join(entity_types)} entities from this text:\n\n{text}",
            "schema": schema,
            "model": model,
            "temperature": 0.1,
        })

    @router.post("/classify")
    async def classify_text(data: dict[str, Any]) -> dict:
        text = data.get("text", "")
        categories = data.get("categories", [])
        model = data.get("model", "llama3.2:1b")

        if not text or not categories:
            raise HTTPException(status_code=400, detail="text and categories required")

        schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": categories},
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
        }

        return await structured_generate({
            "prompt": f"Classify this text into one of: {', '.join(categories)}\n\nText: {text}",
            "schema": schema,
            "model": model,
            "temperature": 0.1,
        })

    @router.post("/summarize")
    async def summarize_structured(data: dict[str, Any]) -> dict:
        text = data.get("text", "")
        model = data.get("model", "llama3.2:1b")
        max_words = data.get("max_words", 50)

        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            },
        }

        return await structured_generate({
            "prompt": f"Summarize this in {max_words} words or less:\n\n{text}",
            "schema": schema,
            "model": model,
            "temperature": 0.3,
        })

    def _format_schema(schema: dict) -> str:
        import json
        return json.dumps(schema, indent=2)

    def _call_ollama(prompt: str, model: str, temperature: float) -> str:
        import json as _json
        from urllib.request import Request, urlopen
        payload = _json.dumps({
            "model": model, "prompt": prompt,
            "stream": False, "options": {"temperature": temperature},
        }).encode()
        req = Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=60) as resp:
            data = _json.loads(resp.read().decode())
            return data.get("response", "").strip()

    def _parse_json_response(raw: str) -> dict | None:
        import json as _json
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return _json.loads(match.group(0))
            except _json.JSONDecodeError:
                pass
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass
        return None

    app.include_router(router)
