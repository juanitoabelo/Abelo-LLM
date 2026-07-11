from __future__ import annotations

import math
import random
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from src.multimodal.planner import ContentPlanner


def _frames_to_arrays(frames: list[Image.Image]) -> list:
    try:
        import numpy as np
        return [np.array(frame) for frame in frames]
    except ImportError:
        return frames  # type: ignore[return-value]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (100, 100, 200)
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_gradient(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    color_top: tuple[int, int, int],
    color_bottom: tuple[int, int, int],
    alpha: int = 255,
) -> None:
    for y in range(height):
        ratio = y / max(1, height - 1)
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * ratio)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * ratio)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * ratio)
        draw.line((0, y, width, y), fill=(r, g, b, alpha))


def add_starfield(draw: ImageDraw.ImageDraw, width: int, height: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    for _ in range(60):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        r = rng.randint(1, 3)
        alpha = rng.randint(60, 200)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, alpha))


def add_particules(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    frame_index: int,
    count: int = 20,
    color: tuple[int, int, int] = (100, 150, 255),
) -> None:
    for i in range(count):
        t = (frame_index * 0.02 + i * 0.5) % (2 * math.pi)
        x = int(width * 0.3 + width * 0.4 * math.sin(t + i))
        y = int(height * 0.3 + height * 0.3 * math.cos(t * 0.7 + i * 1.3))
        r = int(3 + 2 * math.sin(t * 2 + i))
        alpha = int(100 + 100 * math.sin(t + i))
        c = (
            min(255, color[0] + int(50 * math.sin(t))),
            min(255, color[1] + int(50 * math.cos(t))),
            min(255, color[2] + int(50 * math.sin(t * 1.5))),
        )
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(*c, alpha))


async def generate_video_artifact(
    prompt: str,
    output_path: str | Path,
    scene_count: int = 4,
    fps: int = 24,
    seconds_per_scene: float = 3.0,
    storyboard: Optional[list[dict]] = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    planner = ContentPlanner()
    scenes = storyboard or await planner.plan_storyboard(prompt, scene_count)

    width, height = 1280, 720
    frames_per_scene = max(4, int(fps * seconds_per_scene))
    all_frames: list[Image.Image] = []

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_caption = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = font_title
        font_caption = font_title

    for scene_idx, scene in enumerate(scenes):
        palette_raw = scene.get("palette", ["#0a1428", "#3d64ff"])
        palette = [_hex_to_rgb(p) if isinstance(p, str) else tuple(p) for p in palette_raw]
        accent_raw = scene.get("accent", "#6080ff")
        accent = _hex_to_rgb(accent_raw) if isinstance(accent_raw, str) else tuple(accent_raw)
        title = scene.get("title", prompt)[:60]
        description = scene.get("description", "")

        for frame_idx in range(frames_per_scene):
            frame = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(frame)

            progress = frame_idx / max(1, frames_per_scene - 1)
            pulse = 0.5 + 0.5 * math.sin(progress * math.pi * 2)

            draw_gradient(draw, width, height, palette[0], pulse_adjusted(palette[1], pulse))

            add_starfield(draw, width, height, seed=scene_idx)
            add_particules(draw, width, height, frame_idx + scene_idx * 100, color=accent)

            glow_r = int(120 + 100 * pulse)
            glow_alpha = int(30 + 40 * pulse)
            glow_x = int(width * (0.2 + 0.6 * progress))
            glow_y = int(height * 0.4 + 50 * math.sin(progress * math.pi))
            draw.ellipse(
                (glow_x - glow_r, glow_y - glow_r, glow_x + glow_r, glow_y + glow_r),
                fill=(*accent, glow_alpha),
            )

            bounce_y = int(15 * abs(math.sin(progress * math.pi * 3)))
            for i in range(6):
                rx = 100 + i * 140 + int(20 * math.sin(progress * math.pi * 2 + i))
                ry = 420 + bounce_y + i * 20
                rw = 160 + i * 10
                rh = 4 + pulse * 4
                draw.rounded_rectangle(
                    (rx, ry, rx + rw, ry + int(rh)),
                    radius=6 if rh > 3 else 2,
                    fill=(*accent, int(40 + 80 * pulse * (1 - i / 6))),
                )

            title_lines = textwrap.wrap(title, width=35)
            y_text = 60
            for line in title_lines[:2]:
                draw.text((width // 2 - 150, y_text), line, fill="white", font=font_title)
                y_text += 42

            if description:
                desc_lines = textwrap.wrap(description, width=50)
                for i, line in enumerate(desc_lines[:2]):
                    draw.text((width // 2 - 200, 160 + i * 30), line, fill=(200, 220, 255, 200), font=font_body)

            scene_label = f"Scene {scene_idx + 1} of {len(scenes)}"
            draw.text((40, height - 40), scene_label, fill=(180, 200, 230, 180), font=font_caption)
            draw.text((width - 200, height - 40), f"my_custom_llm", fill=(180, 200, 230, 120), font=font_caption)

            all_frames.append(frame)

    frame_arrays = _frames_to_arrays(all_frames)
    suffix = path.suffix.lower()

    if suffix in {".mp4", ".mov", ".mkv"}:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(str(path), frame_arrays, fps=fps, codec="libx264")
        except Exception:
            fallback_save_gif(all_frames, path, fps)
    else:
        fallback_save_gif(all_frames, path, fps)

    return path


def fallback_save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=max(40, int(1000 / max(1, fps))),
        loop=0,
    )


def pulse_adjusted(color: tuple[int, int, int], pulse: float) -> tuple[int, int, int]:
    return (
        min(255, int(color[0] * (0.8 + 0.2 * pulse))),
        min(255, int(color[1] * (0.8 + 0.2 * pulse))),
        min(255, int(color[2] * (0.8 + 0.2 * pulse))),
    )
