# Session Memory — July 12, 2026

## Project: Winner Model (my_custom_llm)

A fully local, independent AI system with zero subscription or API dependency.

---

## What Was Done This Session

### 1. Streaming Bug Fix
- **Problem:** `src/llm/ollama.py` — `_stream_sync()` collected the entire Ollama response into a list before yielding. User saw no output until generation was 100% complete.
- **Fix:** Replaced with `_stream_async()` — uses a background thread + `asyncio.Queue` to yield tokens in real-time as they arrive from Ollama.
- **Files:**
  - `src/llm/ollama.py` — new `_stream_async()` method, `_SENTINEL` module constant
  - Both `generate()` and `chat()` methods now use `_stream_async` for streaming paths

### 2. LocalModelBackend Wired In
- **Problem:** `src/llm/local_model.py` existed but was never used. Router was hardcoded to Ollama only.
- **Fix:** `src/llm/router.py` now imports `LocalModelBackend`, checks `ENABLE_CUSTOM_MODEL` setting, routes to custom model when enabled.
- **Files:**
  - `src/llm/router.py` — `LLMRouter._active_backend()` method, dual backend support
  - `src/server/routes/chat.py` — changed health check from `backends.get("ollama")` to `any(backends.values())`
  - `src/server/routes/generate.py` — same
  - `src/server/routes/models.py` — now returns both Ollama and custom models
  - `src/llm/local_model.py` — now streams tokens via `_generate_tokens()` instead of yielding full result at once
  - `src/inference.py` — added `generate_text_stream()` and `_generate_tokens()` generator

### 3. Frontend Command System
- Chat now supports: `/video <desc>`, `/image <desc>`, `/code <desc>`
- Detects commands, routes to `/api/generate/artifact`
- Renders video player, images, and code inline in chat
- `formatContent()` renders `<video>` and `<img>` HTML tags as React components
- Shows "Generating..." indicator with label

### 4. Video Generation Enhanced
- **Problem:** `src/multimodal/video.py` was an abstract gradient animation — not what user wanted.
- **Fix:** Rewrote with LLM-generated storyboard, cinematic transitions (fade, slide, zoom), particle systems, floating shapes, grid lines, radial glow, futuristic frame corners, animated text overlays.
- **Note:** This is still **animated slideshow**, not realistic AI video. Realistic video needs GPU.
- Added `generate_video_from_description()` — uses LLM to auto-generate storyboard from prompt.
- **Files:**
  - `src/multimodal/video.py` — complete rewrite
  - `src/multimodal/__init__.py` — exports `generate_video_from_description`

### 5. Model Configs Expanded
- Added `medium`, `large`, `xl`, `winner-1b`, `winner-7b`, `winner-70b` configs
- **File:** `configs/model_config.py`

### 6. Fixed Package Installation
- **Problem:** `pyproject.toml` used `setuptools.backends._legacy` which required setuptools >=68, but system had 58.
- **Fix:** Changed build backend to `setuptools.build_meta`
- Now `llm` CLI command works (`llm create --mode video -o out.mp4 "prompt"`)

### 7. Fixed Packaging
- `pyproject.toml`: Changed `setuptools.backends._legacy` → `setuptools.build_meta`
- `.env.example`: Added Winner Model config settings (`ENABLE_CUSTOM_MODEL`, etc.)
- `settings.py`: Added more available models
- `pyproject.toml`: Build backend fix

---

## Current Architecture

```
Browser (Next.js :3000) → FastAPI (:8000)
  ├─ /api/chat → LLMRouter → OllamaBackend (or LocalModelBackend)
  ├─ /api/generate/artifact → multimodal generators
  │   ├─ video: generate_video_from_description() / generate_video_artifact()
  │   ├─ image: generate_image_artifact()  (PIL gradient)
  │   ├─ code: generate_code_artifact()
  │   ├─ audio: generate_audio_artifact()
  │   └─ text: generate_text_artifact()
  └─ /api/models → backend health & model listing
```

---

## Current State of Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Chat (text) | ✅ Working | Ollama backend, SSE streaming, real-time tokens |
| Code generation | ✅ Working | LLM generates code via chat |
| Image generation | ⚠️ Basic | PIL gradient images — needs GPU for real AI images |
| Video generation | ⚠️ Slideshow | Animated transitions/shapes — needs GPU for realistic video |
| Audio generation | ❌ Placeholder | `generate_audio_artifact()` exists but needs GPU models |
| Custom model (Winner) | 🟡 Wired in | Router supports it, needs trained checkpoint |

---

## Hardware Goal

**Target:** Used RTX 3090 (24GB VRAM) — $700-900

Single card unlocks:
- **Wan 2.2** — 720p AI video from text
- **HunyuanVideo 1.5** — 720p video
- **CogVideoX 5B** — 1440×960 video
- **LTX-2.3** — 4K video + audio, up to 20s clips
- **Stable Diffusion / FLUX** — Pro-quality images
- **Qwen3-TTS** — Voice cloning & audio
- **Model fine-tuning & training**
- **Full Winner Model training at larger scales**

See `HARDWARE_REQUIREMENTS.md` for full breakdown.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `ROADMAP.md` | Winner Model vision, phases, milestones |
| `HARDWARE_REQUIREMENTS.md` | GPU/hardware build guide with pricing |
| `configs/model_config.py` | All model sizes (micro → winner-70b) |
| `src/llm/router.py` | Backend router (Ollama + Local) |
| `src/llm/ollama.py` | Ollama backend with real-time streaming |
| `src/llm/local_model.py` | Custom model backend |
| `src/inference.py` | Model inference with token streaming |
| `src/multimodal/video.py` | Video generation (enhanced slideshow) |
| `src/server/routes/chat.py` | Chat API endpoint |
| `src/server/routes/generate.py` | Generation endpoints |
| `frontend/src/app/page.tsx` | Main chat UI with /video, /image, /code |

---

## Next Time Priorities

1. **When GPU arrives:** Install drivers, enable GPU inference, drop in video/image/audio models
2. **Train Winner Model:** Start with `base` or `medium` config, train on quality dataset
3. **Fine-tune open models:** Custom behavior without training from scratch
4. **Video pipeline:** Wire LTX-2.3 or Wan 2.2 into `/video` command for realistic video+audio
5. **Image pipeline:** Wire FLUX or SD into `/image` command
