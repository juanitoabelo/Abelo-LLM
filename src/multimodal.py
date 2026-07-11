from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from src.inference import generate_text as generate_model_text
from src.inference import load_model, load_tokenizer


DEFAULT_CHECKPOINT = "checkpoints/tiny_transformer_best.pt"
DEFAULT_TOKENIZER = "checkpoints/tokenizer.json"


def _safe_text(prompt: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    return cleaned if cleaned else fallback


def _load_local_generator() -> tuple[Optional[object], Optional[object]]:
    try:
        tokenizer = load_tokenizer(DEFAULT_TOKENIZER)
        model = load_model(DEFAULT_CHECKPOINT)
        return tokenizer, model
    except Exception:
        return None, None


def _build_prompt_response(prompt: str) -> str:
    tokenizer, model = _load_local_generator()
    if tokenizer is not None and model is not None:
        try:
            return generate_model_text(prompt, tokenizer, model, max_new_tokens=24, temperature=0.8, top_k=25, top_p=0.95)
        except Exception:
            pass

    prompt_text = prompt.strip() or "Create something useful"
    return (
        f"Here is a concise plan for the request: {prompt_text}. "
        "Break the work into goals, required assets, and a simple execution path."
    )


def classify_request(prompt: str) -> str:
    text = (prompt or "").lower()
    if any(keyword in text for keyword in ["video", "cinematic", "promo", "trailer", "animation", "motion"]):
        return "video"
    if any(keyword in text for keyword in ["infographic", "chart", "timeline", "diagram", "visual summary"]):
        return "infographic"
    if any(keyword in text for keyword in ["image", "poster", "illustration", "visual art", "banner", "cover", "poster design", "artwork"]):
        return "image"
    if any(keyword in text for keyword in ["code", "python", "javascript", "typescript", "html", "css", "script", "function", "api", "cli", "app"]):
        return "code"
    return "text"


def generate_text_artifact(prompt: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _build_prompt_response(prompt)
    path.write_text(content, encoding="utf-8")
    return path


def generate_code_artifact(prompt: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    description = _safe_text(prompt, "build a helpful tool")
    language = "python"
    if re.search(r"\b(js|javascript|node|web|html|css)\b", description, re.I):
        language = "javascript"

    if language == "python":
        code = f'''"""Generated from prompt: {description}"""

from pathlib import Path


def main() -> None:
    print("Hello from the generated assistant workflow")
    print("Prompt:", {description!r})


if __name__ == "__main__":
    main()
'''
    else:
        code = f'''// Generated from prompt: {description}

function main() {{
  console.log("Hello from the generated assistant workflow");
}}

main();
'''

    path.write_text(code, encoding="utf-8")
    return path


def generate_image_artifact(prompt: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 900
    image = Image.new("RGB", (width, height), color=(12, 20, 35))
    draw = ImageDraw.Draw(image)

    for idx in range(5):
        color = ((idx * 43) % 255, (idx * 71) % 255, (idx * 101) % 255)
        draw.ellipse((80 + idx * 110, 60 + idx * 40, 380 + idx * 110, 360 + idx * 40), fill=color)

    draw.rectangle((80, 430, width - 80, height - 80), fill=(24, 32, 53))
    draw.rectangle((120, 470, width - 120, height - 120), fill=(45, 65, 95))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    title = _safe_text(prompt, "creative concept")
    title_lines = textwrap.wrap(title, width=30)
    y = 120
    for line in title_lines[:3]:
        draw.text((120, y), line, fill="white", font=font)
        y += 46

    draw.text((120, 240), "Multimodal concept renderer", fill=(228, 243, 255), font=font)
    image.save(path)
    return path


def generate_infographic_artifact(prompt: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    title = _safe_text(prompt, "AI concept overview")
    words = re.findall(r"[A-Za-z0-9]+", title)
    if len(words) > 6:
        words = words[:6]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">',
        '<rect width="100%" height="100%" fill="#07111f"/>',
        f'<text x="80" y="120" fill="#ffffff" font-size="42" font-family="Arial">{title}</text>',
        '<rect x="80" y="170" width="1240" height="620" rx="28" fill="#0f223c" stroke="#4fc3f7" stroke-width="3"/>',
    ]

    sections = [
        ("Core idea", words[0] if words else "Concept"),
        ("Key value", "clarity and momentum"),
        ("Audience", "builders and creators"),
        ("Outcome", "faster execution"),
    ]

    for index, (label, value) in enumerate(sections):
        x = 120 + (index % 2) * 560
        y = 220 + (index // 2) * 220
        svg_parts.append(f'<rect x="{x}" y="{y}" width="480" height="140" rx="20" fill="#18344f"/>')
        svg_parts.append(f'<text x="{x + 24}" y="{y + 48}" fill="#74d4ff" font-size="24">{label}</text>')
        svg_parts.append(f'<text x="{x + 24}" y="{y + 98}" fill="#ffffff" font-size="28">{value}</text>')

    svg_parts.append('</svg>')
    path.write_text("\n".join(svg_parts), encoding="utf-8")
    return path


def generate_video_artifact(prompt: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 800, 450
    frames = []
    for index in range(6):
        frame = Image.new("RGB", (width, height), color=(7, 18, 38))
        draw = ImageDraw.Draw(frame)
        for offset in range(4):
            color = ((index * 30 + offset * 50) % 255, 50 + offset * 30, 120 + offset * 20)
            draw.rectangle((40 + offset * 80, 50 + index * 20, 760 - offset * 80, 400 - index * 20), outline=color, width=4)
        draw.text((60, 180), _safe_text(prompt, "promo storyboard"), fill="white")
        frames.append(frame)

    frames[0].save(path, save_all=True, append_images=frames[1:], duration=180, loop=0)
    return path


def generate_artifact(prompt: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = classify_request(prompt)

    if mode == "code":
        return generate_code_artifact(prompt, path)
    if mode == "image":
        return generate_image_artifact(prompt, path)
    if mode == "infographic":
        return generate_infographic_artifact(prompt, path)
    if mode == "video":
        return generate_video_artifact(prompt, path)

    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return generate_image_artifact(prompt, path)
    if path.suffix.lower() == ".svg":
        return generate_infographic_artifact(prompt, path)
    if path.suffix.lower() == ".gif":
        return generate_video_artifact(prompt, path)
    if path.suffix.lower() == ".py":
        return generate_code_artifact(prompt, path)
    return generate_text_artifact(prompt, path)
