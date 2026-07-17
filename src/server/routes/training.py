from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi import Request as FastAPIRequest


def register_training_routes(app: "FastAPI") -> None:
    from fastapi import APIRouter, HTTPException, BackgroundTasks
    import json
    from pathlib import Path

    from src.training.data import DatasetBuilder
    from src.training.distill import DistillationTrainer
    from src.training.lora import LoRATrainer

    router = APIRouter(prefix="/api/training", tags=["training"])

    _jobs: dict[str, dict] = {}

    @router.post("/dataset/build")
    async def build_dataset(data: dict[str, Any]) -> dict:
        source = data.get("source")
        source_type = data.get("source_type", "text_files")
        output_name = data.get("output_name", "train")
        format_type = data.get("format", "raw")

        builder = DatasetBuilder()
        if source_type == "text_files":
            samples = builder.from_text_files(source, format=format_type)
        elif source_type == "jsonl":
            samples = builder.from_jsonl(source)
        elif source_type == "qa_pairs":
            samples = builder.from_qa_pairs(source, format=format_type)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source_type: {source_type}")

        if data.get("augment_with_ollama"):
            model = data.get("augment_model", "llama3.2:1b")
            samples = builder.augment_with_ollama(samples, model=model)

        path = builder.convert_to_jsonl(samples, output_name)
        return {
            "status": "ok",
            "samples": len(samples),
            "tokens": builder.count_tokens(samples),
            "path": str(path),
        }

    @router.post("/distill/generate")
    async def generate_distillation(data: dict[str, Any]) -> dict:
        trainer = DistillationTrainer(
            teacher_model=data.get("teacher_model", "qwen3.5:latest"),
            output_dir=data.get("output_dir", "data/distillation"),
        )
        method = data.get("method", "texts")
        if method == "texts":
            samples = trainer.generate_dataset(
                seed_texts=data["seed_texts"],
                num_samples=data.get("num_samples", 100),
                temperature=data.get("temperature", 0.7),
            )
        elif method == "qa":
            samples = trainer.generate_qa_pairs(
                topics=data["topics"],
                num_pairs=data.get("num_pairs", 50),
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

        path = trainer.save_jsonl(samples, data.get("output_name", "distillation_data"))
        return {
            "status": "ok",
            "samples": len(samples),
            "path": str(path),
        }

    @router.post("/distill/run")
    async def run_distillation(data: dict[str, Any]) -> dict:
        trainer = DistillationTrainer(
            teacher_model=data.get("teacher_model", "qwen3.5:latest"),
            output_dir=data.get("output_dir", "data/distillation"),
        )
        result = trainer.run_distillation(
            data_path=data["data_path"],
            student_model=data.get("student_model", "llama3.2:1b"),
            output_name=data.get("output_name", "distilled_model"),
            epochs=data.get("epochs", 3),
        )
        return {"status": "ok", **result}

    @router.post("/lora/prepare")
    async def prepare_lora(data: dict[str, Any]) -> dict:
        trainer = LoRATrainer(backend=data.get("backend", "auto"))
        path = trainer.prepare_data(
            jsonl_path=data["data_path"],
            output_dir=data.get("output_dir", "data/lora"),
        )
        return {"status": "ok", "path": str(path)}

    @router.post("/lora/train")
    async def train_lora(data: dict[str, Any]) -> dict:
        trainer = LoRATrainer(backend=data.get("backend", "auto"))
        result = trainer.train(
            base_model=data["base_model"],
            data_path=data["data_path"],
            output_dir=data.get("output_dir", "data/lora"),
            lora_rank=data.get("rank", 16),
            epochs=data.get("epochs", 3),
            learning_rate=data.get("lr", 2e-4),
            batch_size=data.get("batch_size", 4),
        )
        return {"status": "ok", **result}

    @router.get("/dataset/status")
    async def dataset_status() -> dict:
        paths = list(Path("data/training").glob("*.jsonl"))
        total_tokens = 0
        datasets = []
        for p in paths:
            tokens = sum(len(line) // 4 for line in p.read_text().split("\n") if line.strip())
            total_tokens += tokens
            datasets.append({"name": p.stem, "path": str(p), "tokens": tokens})
        return {"datasets": datasets, "total_tokens": total_tokens}

    app.include_router(router)
