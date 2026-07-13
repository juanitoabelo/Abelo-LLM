from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import click

from src.config.settings import get_settings
from src.llm.router import LLMRouter
from src.multimodal import (
    ContentPlanner,
    generate_code_artifact,
    generate_image_artifact,
    generate_text_artifact,
    generate_video_artifact,
    generate_video_from_description,
    generate_audio_artifact,
)


@click.group()
@click.version_option(version="1.0.0")
def cli() -> None:
    """my_custom_llm - Full-featured local LLM tool with multimodal generation."""
    pass


@cli.command()
@click.option("--model", default=None, help="Ollama model name")
@click.option("--system", default=None, help="System prompt")
@click.option("--temperature", type=float, default=None, help="Sampling temperature")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
@click.argument("prompt", nargs=-1, required=False)
def chat(model: Optional[str], system: Optional[str], temperature: Optional[float], no_stream: bool, prompt: tuple[str, ...]) -> None:
    """Interactive chat with the LLM."""
    from src.cli.chat import run_chat
    initial = " ".join(prompt) if prompt else None
    asyncio.run(run_chat(model=model, system=system, temperature=temperature, stream=not no_stream, initial_prompt=initial))


@cli.command()
@click.option("--model", default=None, help="Ollama model name")
@click.option("--temperature", type=float, default=None, help="Sampling temperature")
@click.option("--max-tokens", type=int, default=None, help="Maximum tokens to generate")
@click.option("--output", "-o", default=None, help="Output file path")
@click.argument("prompt")
def generate(model: Optional[str], temperature: Optional[float], max_tokens: Optional[int], output: Optional[str], prompt: str) -> None:
    """Generate text from a prompt."""
    from rich import print as rprint
    from rich.text import Text

    async def _run() -> None:
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            parts = []
            llm = LLMRouter()
            async for chunk in llm.generate(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, stream=False):
                parts.append(chunk)
            path.write_text("".join(parts), encoding="utf-8")
            rprint(f"[green]Output saved to:[/] {path}")
            return

        llm = LLMRouter()
        first = True
        async for chunk in llm.generate(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, stream=True):
            if first:
                rprint()
                first = False
            print(chunk, end="", flush=True)
        print()

    asyncio.run(_run())


@cli.command()
@click.option("--mode", type=click.Choice(["auto", "text", "image", "video", "code", "audio", "infographic"]), default="auto", help="Output mode")
@click.option("--output", "-o", required=True, help="Output file path")
@click.option("--scenes", type=int, default=4, help="Number of video scenes")
@click.option("--fps", type=int, default=24, help="Video frames per second")
@click.argument("prompt")
def create(mode: str, output: str, scenes: int, fps: int, prompt: str) -> None:
    """Generate an artifact (image, video, code, audio, text)."""
    from rich import print as rprint

    async def _run() -> None:
        planner = ContentPlanner()
        resolved_mode = mode if mode != "auto" else planner.classify_request(prompt)
        output_path = Path(output)

        rprint(f"[bold blue]Generating {resolved_mode}...[/]")

        if resolved_mode == "image":
            if output_path.suffix not in {".png", ".jpg", ".jpeg"}:
                output_path = output_path.with_suffix(".png")
            await generate_image_artifact(prompt, output_path)
        elif resolved_mode == "video":
            if output_path.suffix not in {".mp4", ".gif", ".mov"}:
                output_path = output_path.with_suffix(".mp4")
            await generate_video_from_description(prompt, output_path, fps=fps)
        elif resolved_mode == "code":
            await generate_code_artifact(prompt, output_path)
        elif resolved_mode == "audio":
            await generate_audio_artifact(prompt, output_path)
        else:
            await generate_text_artifact(prompt, output_path)

        file_size = output_path.stat().st_size
        rprint(f"[green]Created:[/] {output_path.resolve()} ({_format_size(file_size)})")

    asyncio.run(_run())


@cli.command()
def models() -> None:
    """List available LLM models."""
    from rich import print as rprint
    from rich.table import Table
    from rich.console import Console

    async def _run() -> None:
        llm = LLMRouter()
        backends = await llm.check_backends()

        table = Table(title="Available Backends")
        table.add_column("Backend", style="cyan")
        table.add_column("Status", style="green")

        for name, available in backends.items():
            table.add_row(name.capitalize(), "✓ Available" if available else "✗ Unavailable")

        console = Console()
        console.print(table)

        if backends.get("ollama"):
            models_list = await llm.list_models()
            if models_list:
                model_table = Table(title="Ollama Models")
                model_table.add_column("Name", style="cyan")
                model_table.add_column("Size", style="yellow")
                model_table.add_column("Family", style="green")
                model_table.add_column("Quantization", style="blue")

                for m in models_list:
                    details = m.get("details", {})
                    size_bytes = m.get("size", 0)
                    model_table.add_row(
                        m["name"],
                        _format_size(size_bytes) if size_bytes else "?",
                        details.get("family", "?"),
                        details.get("quantization_level", "?"),
                    )
                console.print(model_table)

    asyncio.run(_run())


@cli.command()
def serve() -> None:
    """Start the FastAPI server."""
    from src.server.app import main
    main()


def _format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    cli()
