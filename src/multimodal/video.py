from __future__ import annotations

import json
import math
import random
import subprocess
import textwrap
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from src.multimodal.planner import ContentPlanner


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (60, 80, 140)
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        min(255, int(a[0] + (b[0] - a[0]) * t)),
        min(255, int(a[1] + (b[1] - a[1]) * t)),
        min(255, int(a[2] + (b[2] - a[2]) * t)),
    )


def draw_gradient(draw: ImageDraw.ImageDraw, width: int, height: int, color_top: tuple, color_bottom: tuple, alpha: int = 255) -> None:
    for y in range(height):
        ratio = y / max(1, height - 1)
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * ratio)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * ratio)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * ratio)
        draw.line((0, y, width, y), fill=(r, g, b, alpha))


def draw_radial_glow(frame: Image.Image, cx: int, cy: int, radius: int, color: tuple, max_alpha: int = 80) -> None:
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    for r in range(radius, 0, -max(1, radius // 30)):
        t = 1 - r / radius
        alpha = int(max_alpha * t * t)
        gdraw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*color, alpha))
    frame.paste(glow, (0, 0), glow)


def add_starfield(draw: ImageDraw.ImageDraw, width: int, height: int, seed: int = 0, count: int = 80) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        r = rng.randint(1, 3)
        alpha = rng.randint(40, 220)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, alpha))


def add_grid_lines(draw: ImageDraw.ImageDraw, width: int, height: int, offset: float = 0, color: tuple = (100, 150, 255, 30)) -> None:
    spacing = 60
    for x in range(0, width + spacing, spacing):
        px = (x + int(offset * 20)) % (width + spacing * 2) - spacing
        draw.line([(px, 0), (px, height)], fill=color, width=1)
    for y in range(0, height + spacing, spacing):
        py = (y + int(offset * 15)) % (height + spacing * 2) - spacing
        draw.line([(0, py), (width, py)], fill=color, width=1)


def add_floating_shapes(draw: ImageDraw.ImageDraw, width: int, height: int, t: float, accent: tuple, count: int = 8) -> None:
    rng = random.Random(int(t * 10))
    for i in range(count):
        cx = int(width * (0.1 + 0.8 * ((math.sin(t * 0.5 + i * 2.1) + 1) / 2)))
        cy = int(height * (0.1 + 0.8 * ((math.cos(t * 0.3 + i * 1.7) + 1) / 2)))
        size = 10 + 25 * (0.5 + 0.5 * math.sin(t + i))
        alpha = int(30 + 60 * (0.5 + 0.5 * math.sin(t * 0.7 + i)))
        shape_type = i % 3
        c = (min(255, accent[0] + int(30 * math.sin(t + i))), min(255, accent[1] + int(30 * math.cos(t * 0.5 + i))), min(255, accent[2] + int(30 * math.sin(t * 0.3 + i * 2))))
        if shape_type == 0:
            draw.ellipse((cx - size, cy - size, cx + size, cy + size), fill=(*c, alpha))
        elif shape_type == 1:
            draw.regular_polygon((cx, cy, size), 6, rotation=int(t * 30 + i * 60), fill=(*c, alpha))
        else:
            draw.rectangle((cx - size, cy - size / 2, cx + size, cy + size / 2), fill=(*c, alpha))


def draw_futuristic_frame(draw: ImageDraw.ImageDraw, width: int, height: int, accent: tuple, t: float) -> None:
    corner_len = 40
    offset = int(5 * math.sin(t * 2))
    for corner in [
        (0, 0, 1, 1),
        (width - 1, 0, -1, 1),
        (0, height - 1, 1, -1),
        (width - 1, height - 1, -1, -1),
    ]:
        cx, cy, dx, dy = corner
        draw.line([(cx, cy + corner_len + offset), (cx, cy), (cx + corner_len + offset, cy)], fill=(*accent, 150), width=2)


def create_scene_frame(
    width: int,
    height: int,
    scene: dict,
    frame_index: int,
    total_frames: int,
    fonts: dict,
) -> Image.Image:
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(frame)

    progress = frame_index / max(1, total_frames - 1)
    t = frame_index * 0.05

    palette_raw = scene.get("palette", ["#0a1428", "#1a3a6a"])
    palette = [_hex_to_rgb(p) if isinstance(p, str) else tuple(p) for p in palette_raw]
    accent_raw = scene.get("accent", "#3d64ff")
    accent = _hex_to_rgb(accent_raw) if isinstance(accent_raw, str) else tuple(accent_raw)

    mood = scene.get("mood", "cinematic")
    color_top = palette[0]
    color_bottom = _lerp_color(palette[-1], accent, 0.3 + 0.3 * math.sin(progress * math.pi))

    draw_gradient(draw, width, height, color_top, color_bottom)
    add_starfield(draw, width, height, seed=frame_index)
    add_grid_lines(draw, width, height, offset=progress, color=(*accent, 15 + int(10 * math.sin(t))))
    add_floating_shapes(draw, width, height, t, accent)
    draw_radial_glow(frame, width // 2, height // 2, 300, accent, max_alpha=30)

    pulse = 0.6 + 0.4 * math.sin(progress * math.pi * 3)
    if mood in ("dramatic", "intense"):
        shake_x = int(4 * math.sin(t * 8))
        shake_y = int(3 * math.cos(t * 7))
        draw_futuristic_frame(draw, width, height, accent, t)
    else:
        shake_x = 0
        shake_y = 0

    title = scene.get("title", "")[:60]
    description = scene.get("description", "")

    y_offset = -30 + 20 * (1 - progress)
    glow_r = int(150 + 80 * pulse)
    draw_radial_glow(frame, width // 2 + shake_x, int(height * 0.25) + shake_y, glow_r, accent, max_alpha=int(25 * pulse))

    if title:
        font_title = fonts.get("title") or ImageFont.load_default()
        title_lines = textwrap.wrap(title, width=30)
        y_text = 60 + y_offset + shake_y
        for line in title_lines[:2]:
            bbox = draw.textbbox((0, 0), line, font=font_title)
            tw = bbox[2] - bbox[0]
            draw.text(((width - tw) // 2 + shake_x, y_text + shake_y), line, fill=(255, 255, 255, int(255 * pulse)), font=font_title)
            y_text += int(bbox[3] - bbox[1]) + 10

    if description:
        font_body = fonts.get("body") or ImageFont.load_default()
        desc_lines = textwrap.wrap(description, width=45)
        y_text_desc = int(height * 0.35) + shake_y
        fade_alpha = int(180 * min(1, progress * 2))
        for i, line in enumerate(desc_lines[:3]):
            bbox = draw.textbbox((0, 0), line, font=font_body)
            tw = bbox[2] - bbox[0]
            alpha = max(0, fade_alpha - i * 30)
            draw.text(((width - tw) // 2 + shake_x, y_text_desc + i * 30 + shake_y), line, fill=(200, 220, 255, alpha), font=font_body)

    scene_label = scene.get("scene_label", "")
    if scene_label:
        font_caption = fonts.get("caption") or ImageFont.load_default()
        draw.text((40, height - 40), scene_label, fill=(180, 200, 230, 160), font=font_caption)

    draw.text((width - 180, height - 40), "Winner Model", fill=(180, 200, 230, 100), font=fonts.get("caption") or ImageFont.load_default())

    return frame


def blend_frames(a: Image.Image, b: Image.Image, t: float, transition: str = "fade") -> Image.Image:
    if transition == "fade":
        return Image.blend(a, b, t)
    elif transition == "slide_left":
        w = a.width
        offset = int(w * t)
        result = Image.new("RGBA", a.size, (0, 0, 0, 255))
        result.paste(b, (offset - w, 0))
        result.paste(a, (offset, 0))
        return result
    elif transition == "slide_up":
        h = a.height
        offset = int(h * t)
        result = Image.new("RGBA", a.size, (0, 0, 0, 255))
        result.paste(b, (0, offset - h))
        result.paste(a, (0, offset))
        return result
    elif transition == "zoom_in":
        scale = 1 + 0.1 * t
        sw, sh = int(a.width * scale), int(a.height * scale)
        zoomed = b.resize((sw, sh), Image.LANCZOS)
        result = Image.new("RGBA", a.size, (0, 0, 0, 255))
        result.paste(zoomed, ((a.width - sw) // 2, (a.height - sh) // 2))
        return Image.blend(a, result, t)
    else:
        return Image.blend(a, b, t)


async def generate_video_artifact(
    prompt: str,
    output_path: str | Path,
    scene_count: int = 4,
    fps: int = 24,
    seconds_per_scene: float = 4.0,
    storyboard: Optional[list[dict]] = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    planner = ContentPlanner()
    scenes = storyboard or await planner.plan_storyboard(prompt, scene_count)

    width, height = 1280, 720
    transition_frames = int(fps * 0.5)
    frames_per_scene = max(8, int(fps * seconds_per_scene))
    all_frames: list[Image.Image] = []

    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]
    fonts = {"title": None, "body": None, "caption": None}
    for fp in font_paths:
        try:
            fonts["title"] = ImageFont.truetype(fp, 36)
            fonts["body"] = ImageFont.truetype(fp, 22)
            fonts["caption"] = ImageFont.truetype(fp, 14)
            break
        except Exception:
            continue
    if fonts["title"] is None:
        fonts["title"] = fonts["body"] = fonts["caption"] = ImageFont.load_default()

    transitions = ["fade", "slide_left", "slide_up", "zoom_in", "fade"]

    for scene_idx, scene in enumerate(scenes):
        scene["scene_label"] = f"Scene {scene_idx + 1} of {len(scenes)}"
        transition = transitions[scene_idx % len(transitions)]

        for frame_idx in range(frames_per_scene):
            frame = create_scene_frame(width, height, scene, frame_idx, frames_per_scene, fonts)
            all_frames.append(frame)

        if scene_idx < len(scenes) - 1:
            next_scene = scenes[scene_idx + 1]
            trans_out = all_frames[-1].copy()
            next_first = create_scene_frame(width, height, next_scene, 0, frames_per_scene, fonts)
            for ti in range(transition_frames):
                t = (ti + 1) / (transition_frames + 1)
                blended = blend_frames(trans_out, next_first, t, transition)
                all_frames.append(blended)

    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".webm"} and _ffmpeg_available():
        _save_mp4(all_frames, path, fps)
    else:
        _save_gif(all_frames, path, fps)

    return path


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=3)
        return True
    except Exception:
        return False


def _save_mp4(frames: list[Image.Image], path: Path, fps: int) -> None:
    import numpy as np
    rgba_frames = [np.array(f.convert("RGBA")) for f in frames]
    rgb_frames = [f[..., :3] for f in rgba_frames]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        import imageio.v2 as imageio
        imageio.mimsave(tmp_path, rgb_frames, fps=fps, codec="libx264", quality=8)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18", str(path)],
            capture_output=True,
            timeout=120,
        )
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        _save_gif(frames, path, fps)


def _save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    frames[0].save(
        str(path),
        save_all=True,
        append_images=frames[1:],
        duration=max(20, int(1000 / max(1, fps))),
        loop=0,
    )


async def generate_video_from_description(prompt: str, output_path: str | Path, fps: int = 24) -> Path:
    planner = ContentPlanner()
    llm = planner.llm

    storyboard_prompt = (
        f"You are a professional video director and storyboard artist. "
        f"For the concept/description: '{prompt}', create exactly 6 distinct scenes. "
        f"For each scene provide:\n"
        f"- title (short, evocative, max 40 chars)\n"
        f"- description (rich visual details, mood, lighting, camera angle, max 120 chars)\n"
        f"- color palette (two hex colors for gradient that match the mood)\n"
        f"- mood (one of: cinematic, dramatic, serene, intense, dreamy, futuristic)\n"
        f"- accent color (one hex color for highlights)\n\n"
        f"Return ONLY a valid JSON array of 6 objects with keys: title, description, palette, mood, accent.\n"
        f"Example: [{{\"title\": \"Scene Title\", \"description\": \"Visual description here\", \"palette\": [\"#000000\", \"#ffffff\"], \"mood\": \"cinematic\", \"accent\": \"#ff0000\"}}]"
    )

    full_response = []
    try:
        async for chunk in llm.generate(prompt=storyboard_prompt, temperature=0.8, max_tokens=512, stream=True):
            full_response.append(chunk)
        text = "".join(full_response)
        import re
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            scenes = json.loads(json_match.group())
            if isinstance(scenes, list) and len(scenes) > 0:
                return await generate_video_artifact(prompt, output_path, scene_count=len(scenes), fps=fps, storyboard=scenes)
    except Exception:
        pass

    return await generate_video_artifact(prompt, output_path, scene_count=6, fps=fps)
