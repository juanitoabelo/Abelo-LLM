from __future__ import annotations

from src.vision.analyzer import VisionAnalyzer


class TestVisionAnalyzer:
    def test_analyzer_init(self) -> None:
        analyzer = VisionAnalyzer()
        assert analyzer is not None
        assert analyzer.model == "gemma4:latest"

    def test_analyzer_custom_model(self) -> None:
        analyzer = VisionAnalyzer(model="llava:latest")
        assert analyzer.model == "llava:latest"

    def test_analyzer_repr(self) -> None:
        analyzer = VisionAnalyzer()
        assert "gemma4" in analyzer.model
