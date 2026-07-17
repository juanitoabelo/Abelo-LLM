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
        merger = ModelMerger(output_dir="/tmp")
        result = merger.merge_lora_adapters("base", ["/tmp/adapter"], method="invalid")
        assert result["status"] == "error"

    def test_merge_gguf_slerp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merger = ModelMerger(output_dir=tmpdir)
            result = merger.merge_gguf(["/tmp/model1.gguf", "/tmp/model2.gguf"], method="slerp")
            assert result["status"] == "script_generated"
            assert result["method"] == "slerp"

    def test_merge_gguf_single_model_error(self) -> None:
        merger = ModelMerger(output_dir="/tmp")
        result = merger.merge_gguf(["/tmp/model1.gguf"], method="slerp")
        assert result["status"] == "error"

    def test_merge_task_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merger = ModelMerger(output_dir=tmpdir)
            result = merger.merge_task_arithmetic("base_model", [("task_vector_1", 0.5), ("task_vector_2", 0.5)])
            assert result["status"] == "script_generated"

    def test_merge_pytorch_few_checkpoints(self) -> None:
        merger = ModelMerger()
        result = merger.merge_pytorch_checkpoints(["/tmp/cp1.pt"], output_path="/tmp/out.pt")
        assert result["status"] == "error"
