from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.multimodal.planner import ContentPlanner
from src.multimodal.image import generate_image_artifact


class TestContentPlanner:
    def test_classify_video(self) -> None:
        planner = ContentPlanner()
        assert planner.classify_request("make a video about space") == "video"
        assert planner.classify_request("cinematic trailer for product") == "video"

    def test_classify_image(self) -> None:
        planner = ContentPlanner()
        assert planner.classify_request("create an image of a cat") == "image"
        assert planner.classify_request("design a poster") == "image"

    def test_classify_code(self) -> None:
        planner = ContentPlanner()
        assert planner.classify_request("write a python function") == "code"
        assert planner.classify_request("build an API") == "code"

    def test_classify_audio(self) -> None:
        planner = ContentPlanner()
        assert planner.classify_request("create audio narration") == "audio"
        assert planner.classify_request("turn this into speech") == "audio"

    def test_classify_text_fallback(self) -> None:
        planner = ContentPlanner()
        assert planner.classify_request("what is the meaning of life") == "text"

    def test_fallback_storyboard(self) -> None:
        planner = ContentPlanner()
        scenes = planner._fallback_storyboard("sunset over mountains", 3)
        assert len(scenes) == 3
        for scene in scenes:
            assert "title" in scene
            assert "palette" in scene
            assert "mood" in scene
            assert "accent" in scene

    def test_fallback_storyboard_eight_scenes(self) -> None:
        planner = ContentPlanner()
        scenes = planner._fallback_storyboard("hello world", 8)
        assert len(scenes) == 8


class TestImageGeneration:
    @pytest.mark.asyncio
    async def test_generate_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test.png"
            result = await generate_image_artifact("A beautiful sunset", output)
            assert result.exists()
            assert result.suffix == ".png"
            assert result.stat().st_size > 100

    @pytest.mark.asyncio
    async def test_generate_with_custom_palette(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test.png"
            result = await generate_image_artifact("neon city", output, palette=["#ff0066", "#00ffcc"])
            assert result.exists()
            assert result.stat().st_size > 100

    @pytest.mark.asyncio
    async def test_generate_with_empty_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test.png"
            result = await generate_image_artifact("", output)
            assert result.exists()
