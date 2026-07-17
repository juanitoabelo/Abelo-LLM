from __future__ import annotations

import pytest

from src.vision.analyzer import VisionAnalyzer


class TestVisionAnalyzer:
    def test_analyzer_init(self) -> None:
        analyzer = VisionAnalyzer()
        assert analyzer is not None

    def test_available_models(self) -> None:
        analyzer = VisionAnalyzer()
        models = analyzer.supported_models
        assert len(models) > 0

    def test_analyzer_repr(self) -> None:
        analyzer = VisionAnalyzer()
        assert "VisionAnalyzer" in repr(analyzer)
