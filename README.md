# my_custom_llm

A full-featured local LLM with RAG, tool calling, persistent memory, agent planning,
guardrails, vision, voice, knowledge graphs, model merging, fine-tuning, and plugins.
Powered by Ollama — no cloud dependencies, no API keys.

## Features

| Category | Capabilities |
|----------|-------------|
| **Chat** | Streaming SSE, multi-model, thinking traces, image attachments |
| **RAG** | Vector store, hybrid search, reranking, web fallback |
| **Tools** | Web search/fetch, calculator, code execution, file I/O, SQL queries, image analysis, memory tools |
| **Memory** | Persistent key-value store, session context, semantic search |
| **Agent** | ReAct planning with self-reflection, multi-step tool execution |
| **Vision** | Multimodal image analysis via Ollama (gemma4, llava) |
| **Voice** | Speech-to-text (whisper) + text-to-speech (pyttsx3/edge-tts) |
| **Knowledge Graph** | Entity extraction, relationship co-occurrence, graph traversal |
| **Training** | Fine-tuning pipeline, LoRA, distillation, quantization |
| **Model Merging** | SLERP, TIES, task arithmetic, LoRA stacking |
| **Auth** | Multi-user JWT authentication |
| **Guardrails** | Content filtering, PII detection, rate limiting |
| **Plugins** | Discover, load, unload plugin modules at runtime |
| **Structured Output** | JSON generation, data extraction, classification, summarization |
| **Observability** | Request tracking, token counting, stats dashboard |
| **Frontend** | Next.js 14 chat UI with dark/light mode, model management |

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai)
- At least one model: `ollama pull qwen3.5:latest`

### Install

```bash
cd my_custom_llm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"      # for development
pip install -e ".[voice]"    # for voice support
pip install -e ".[torch]"    # for model merging
```

### Usage

```bash
# CLI chat
llm chat
llm chat "Explain quantum computing"

# Start web server
llm serve

# Start frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open http://localhost:3000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Streaming chat with SSE |
| `/api/generate/text` | POST | Text generation |
| `/api/generate/artifact` | POST | Generate image/video/code/audio |
| `/api/models` | GET | List models |
| `/api/models/health` | GET | Health check |
| `/api/rag/status` | GET | RAG status |
| `/api/rag/ingest/text` | POST | Ingest text into RAG |
| `/api/rag/ingest/file` | POST | Ingest file into RAG |
| `/api/rag/query` | POST | Query RAG |
| `/api/memory` | GET | List memory keys |
| `/api/memory/remember` | POST | Store fact |
| `/api/memory/recall/{key}` | GET | Recall fact |
| `/api/stats` | GET | Usage statistics |
| `/api/agent/plan` | POST | Create agent plan |
| `/api/agent/plan/{id}` | GET | Get plan |
| `/api/agent/plan/{id}/execute` | POST | Execute plan |
| `/api/auth/register` | POST | Register user |
| `/api/auth/login` | POST | Login |
| `/api/auth/me` | GET | Current user |
| `/api/voice/stt` | POST | Speech-to-text |
| `/api/voice/tts` | POST | Text-to-speech |
| `/api/branch/create` | POST | Create conversation branch |
| `/api/branch/list/{session_id}` | GET | List branches |
| `/api/branch/{branch_id}` | GET | Get branch |
| `/api/branch/templates` | GET | List prompt templates |
| `/api/branch/templates/save` | POST | Save prompt template |
| `/api/branch/templates/apply` | POST | Apply template with variables |
| `/api/training/dataset/build` | POST | Build dataset |
| `/api/training/distill/generate` | POST | Generate distillation data |
| `/api/training/lora/train` | POST | Run LoRA fine-tuning |
| `/api/structured/generate` | POST | Structured JSON generation |
| `/api/structured/extract` | POST | Data extraction |
| `/api/structured/classify` | POST | Text classification |
| `/api/plugins` | GET | List plugins |
| `/api/plugins/load` | POST | Load plugin |
| `/api/upload` | POST | Upload file |

## Architecture

```
src/
├── agent/          # ReAct planner + execution
├── auth/           # JWT multi-user auth
├── cli/            # Rich CLI (click)
├── config/         # Settings (env-based)
├── context/        # Context window management
├── guard/          # Content/PII filters
├── knowledge_graph/# Entity extraction & traversal
├── llm/            # Ollama + local model backends
├── mcp/            # Model Context Protocol
├── memory/         # Persistent key-value store
├── merge/          # Model merging (SLERP, TIES)
├── monitor/        # Usage tracking & stats
├── multimodal/     # Image/video/audio/code generation
├── plugins/        # Runtime plugin system
├── rag/            # Vector store, hybrid search, reranking
├── server/         # FastAPI app + 14 route modules
├── tools/          # 12 tool implementations
├── training/       # Fine-tuning, distillation, LoRA
├── vision/         # Multimodal image analysis
├── voice/          # STT/TTS interface
├── model.py        # TinyTransformer architecture
├── tokenizer.py    # BPE tokenizer
└── train.py        # Training loop
```

## Architecture Diagram

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Frontend   │────▶│  FastAPI     │────▶│  Ollama API     │
│  (Next.js)  │◀────│  Server      │◀────│  (LLM models)   │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────┴───────┐
                    │  LLMRouter   │
                    └──┬──┬──┬──┬─┘
                       │  │  │  │
              ┌────────┘  │  │  └──────────┐
              ▼           ▼  ▼             ▼
        ┌─────────┐ ┌─────────┐ ┌──────────────────┐
        │   RAG   │ │ Memory  │ │ Agent Planner   │
        │ Vector  │ │ Store   │ │ ReAct + Tools    │
        │ Store   │ │         │ │                  │
        └─────────┘ └─────────┘ └──────────────────┘
```

## Docker

```bash
docker-compose up --build
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install          # Install git hooks

pytest                      # Run tests
ruff check src/             # Lint
mypy src/                   # Type check
```

## Training

```bash
# Train from scratch
python -m src.train --data-dir ./data --model-size base --epochs 50

# LoRA fine-tune
python scripts/finetune_lora.py --base-model llama3.2:1b

# Distillation
python scripts/train_pipeline.py --teacher qwen3.5 --student llama3.2:1b

# Self-improvement loop
python scripts/self_improve.py --rounds 3 --model llama3.2:1b

# Model merging
python -c "from src.merge.merger import ModelMerger; m=ModelMerger(); print(m.merge_gguf(['model1.gguf','model2.gguf'], method='slerp'))"
```

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_DEFAULT_MODEL` | `llama3.2:1b` | Default model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8000` | Server port |
| `ENABLE_RAG` | `true` | Enable RAG |
| `ENABLE_TOOLS` | `true` | Enable tools |
| `ENABLE_MEMORY` | `true` | Enable memory |
| `ENABLE_AGENT` | `true` | Enable agent |
| `ENABLE_VISION` | `true` | Enable vision |
| `ENABLE_VOICE` | `true` | Enable voice |
| `ENABLE_GUARDRAILS` | `true` | Enable guardrails |
| `ENABLE_THINKING` | `true` | Enable thinking traces |
| `RATE_LIMIT_MAX` | `30` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (s) |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | CORS origins |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1,...` | Trusted hosts |
