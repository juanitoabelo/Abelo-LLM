from __future__ import annotations

import asyncio
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from src.llm.router import LLMRouter

console = Console()


async def run_chat(
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    stream: bool = True,
    initial_prompt: Optional[str] = None,
) -> None:
    llm = LLMRouter()

    backends = await llm.check_backends()
    if not backends.get("ollama"):
        console.print("[red]Error:[/] Ollama is not running. Start it with: ollama serve")
        return

    models_list = await llm.list_models()
    if models_list:
        model_name = model or models_list[0]["name"]
    else:
        model_name = model or "qwen3.5:latest"

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]my_custom_llm Chat[/]\n"
        f"Model: [green]{model_name}[/] | "
        f"Type [bold]/exit[/] to quit, [bold]/help[/] for commands",
        border_style="blue",
    ))
    console.print()

    history: list[dict] = []

    if initial_prompt:
        console.print(f"[bold blue]You:[/] {initial_prompt}")
        await _process_message(llm, initial_prompt, model_name, system, temperature, stream, history)

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/]")
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in ("/exit", "/quit"):
            console.print("[yellow]Goodbye![/]")
            break

        if user_input.strip().lower() == "/clear":
            history.clear()
            console.clear()
            continue

        if user_input.strip().lower() == "/help":
            console.print(Panel.fit(
                "[bold]Commands:[/]\n"
                "/exit, /quit - Exit the chat\n"
                "/clear - Clear conversation history\n"
                "/help - Show this help",
                border_style="yellow",
            ))
            continue

        if user_input.strip().lower().startswith("/model "):
            model_name = user_input.strip().split(" ", 1)[1]
            console.print(f"[green]Switched to model: {model_name}[/]")
            continue

        await _process_message(llm, user_input, model_name, system, temperature, stream, history)


async def _process_message(
    llm: LLMRouter,
    user_message: str,
    model: str,
    system: Optional[str],
    temperature: Optional[float],
    stream: bool,
    history: list[dict],
) -> None:
    messages = llm.format_chat_messages(system, history, user_message)

    if stream:
        response_text = ""
        spinner = Spinner("dots", text="Thinking...")
        with Live(spinner, refresh_per_second=10, console=console):
            async for chunk in llm.chat(messages=messages, model=model, temperature=temperature, stream=True):
                response_text += chunk
                spinner.update(Text.from_markup(f"[dim]{response_text[-40:]}[/]"))

        console.print()
        md = Markdown(response_text)
        console.print(Panel(md, border_style="green"))
    else:
        response_parts = []
        async for chunk in llm.chat(messages=messages, model=model, temperature=temperature, stream=False):
            response_parts.append(chunk)
        response_text = "".join(response_parts)
        md = Markdown(response_text)
        console.print(Panel(md, border_style="green"))

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response_text})
