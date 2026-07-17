from __future__ import annotations

import tempfile
from pathlib import Path

from src.merge.merger import ModelMerger


class TestModelMerger:
    def test_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merger = ModelMerger(output_dir=tmpdir)
            assert merger.output_dir == Path(tmpdir)

    def test_merge_lora_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merger = ModelMerger(output_dir=tmpdir)
            result = merger.merge_lora_adapters("llama3.2:1b", ["/tmp/adapter1", "/tmp/adapter2"], method="stack")
            assert result["status"] == "created"
            assert result["method"] == "stack"

    def test_merge_lora_unknown_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merger = ModelMerger(output_dir=tmpdir)
            result = merger.merge_lora_adapters("base", ["/tmp/adapter"], method="invalid")
            assert result["status"] == "error"

    def test_merge_gguf_slerp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merger = ModelMerger(output_dir=tmpdir)
            result = merger.merge_gguf(["/tmp/model1.gguf", "/tmp/model2.gguf"], method="slerp")
            assert result["status"] == "script_generated"

    def test_merge_gguf_single_model_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merger = ModelMerger(output_dir=tmpdir)
            result = merger.merge_gguf(["/tmp/model1.gguf"], method="slerp")
            assert result["status"] == "error"

    def test_merge_task_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merger = ModelMerger(output_dir=tmpdir)
            result = merger.merge_task_arithmetic("base_model", [("tv1", 0.5), ("tv2", 0.5)])
            assert result["status"] == "script_generated"
