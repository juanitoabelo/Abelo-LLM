# my_custom_llm

A full-featured local LLM tool with multimodal generation capabilities. Runs entirely on your machine using Ollama for inference — no cloud dependencies, no API keys needed.

## Features

- **Interactive Chat** — Rich CLI and web UI with streaming responses
- **Multimodal Generation** — Images, videos, code, audio, infographics
- **Local Inference** — Powered by Ollama (supports Qwen, Gemma, Llama, Mistral, and 100+ models)
- **REST API** — FastAPI server with WebSocket streaming for chat
- **Custom Transformer** — Train your own tiny transformer from scratch (included)
- **Docker Support** — Full containerized deployment

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) (for LLM inference)
- At least one model: `ollama pull qwen3.5:latest` (or gemma4, llama3.2, etc.)

### Install

```bash
# Clone and enter the project
cd my_custom_llm

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Usage

**CLI Chat:**
```bash
llm chat
```

**CLI Chat with initial prompt:**
```bash
llm chat "Explain quantum computing in simple terms"
```

**Generate text:**
```bash
llm generate "Write a poem about AI" --temperature 0.8
```

**Generate an artifact:**
```bash
# Auto-detect mode from prompt
llm create -o output.mp4 "Create a cinematic video about space exploration"

# Specify mode explicitly
llm create --mode image -o artwork.png "A futuristic city at sunset"
llm create --mode code -o app.py "A REST API with FastAPI"
```

**Start the web server:**
```bash
llm serve
# Or directly: python -m src.server.app
```

**List available models:**
```bash
llm models
```

### Web UI

Start the server, then in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with streaming SSE response |
| `/api/generate/text` | POST | Generate text from prompt |
| `/api/generate/artifact` | POST | Generate image/video/code/audio |
| `/api/models` | GET | List available models |
| `/api/models/health` | GET | Backend health check |

## Architecture

```
src/
├── config/          # Settings & configuration
├── llm/             # LLM abstraction layer (Ollama + custom model)
├── multimodal/      # Image, video, code, audio, text generation
├── server/          # FastAPI server with REST + SSE streaming
├── cli/             # Rich CLI with interactive chat
├── model.py         # Custom tiny transformer architecture
├── tokenizer.py     # BPE tokenizer implementations
├── train.py         # Training loop
└── inference.py     # Text generation with sampling
```

## Development

```bash
# Run tests
pytest

# Type checking
pip install mypy && mypy src/

# Linting
pip install ruff && ruff check src/
```

## Docker

```bash
docker-compose up --build
```

## Training Your Own Model

```bash
python -m src.train --data-dir ./data --model-size base --epochs 50 --batch-size 32
```
