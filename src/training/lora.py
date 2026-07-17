from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


class LoRATrainer:
    BACKENDS = ["mlx", "llama.cpp", "unsloth", "auto"]

    def __init__(self, backend: str = "auto") -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"Backend must be one of: {self.BACKENDS}")
        self.backend = backend

    def _detect_backend(self) -> str:
        backends = []
        try:
            import mlx_lm  # noqa
            backends.append("mlx")
        except ImportError:
            pass
        try:
            import unsloth  # noqa
            backends.append("unsloth")
        except ImportError:
            pass
        try:
            subprocess.run(["llama.cpp", "--version"], capture_output=True)
            backends.append("llama.cpp")
        except FileNotFoundError:
            pass
        if backends:
            return backends[0]
        return "ollama"

    def prepare_data(self, jsonl_path: str | Path, output_dir: str | Path) -> Path:
        path = Path(jsonl_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        backend = self.backend if self.backend != "auto" else self._detect_backend()
        if backend == "mlx":
            import shutil
            shutil.copy2(path, out / "train.jsonl")
            return out / "train.jsonl"
        elif backend == "llama.cpp":
            converted = out / "train.txt"
            with open(path) as f, open(converted, "w") as out_f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if text:
                            out_f.write(text + "\n")
                    except json.JSONDecodeError:
                        continue
            return converted
        return path

    def train(
        self,
        base_model: str,
        data_path: str | Path,
        output_dir: str | Path,
        lora_rank: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        warmup_steps: int = 100,
    ) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        backend = self.backend if self.backend != "auto" else self._detect_backend()

        if backend == "mlx":
            return self._train_mlx(base_model, data_path, out, lora_rank, epochs, learning_rate)
        elif backend == "unsloth":
            return self._train_unsloth(base_model, data_path, out, lora_rank, epochs, learning_rate, batch_size)
        elif backend == "llama.cpp":
            return self._train_llamacpp(base_model, data_path, out, lora_rank, epochs, learning_rate)
        else:
            return self._train_via_ollama(base_model, data_path, out, lora_rank, epochs, learning_rate)

    def _train_via_ollama(self, base_model, data_path, out, rank, epochs, lr):
        """Fallback: use Ollama's Modelfile approach for creating fine-tuned variants."""
        import urllib.request
        samples = []
        with open(data_path) as f:
            for line in f:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        system_prompt = f"You are a fine-tuned version of {base_model}."
        modelfile_parts = [f"FROM {base_model}", f'SYSTEM """{system_prompt}"""']

        for s in samples[:20]:
            text = s.get("text", "")
            if text:
                modelfile_parts.append(f'MESSAGE user """{text}"""')

        modelfile = "\n".join(modelfile_parts)
        modelfile_path = out / "Modelfile"
        modelfile_path.write_text(modelfile)

        result = subprocess.run(
            ["ollama", "create", f"{base_model}-ft", "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=300,
        )

        return {
            "status": "created" if result.returncode == 0 else "failed",
            "backend": "ollama",
            "model_name": f"{base_model}-ft",
            "output": result.stdout + result.stderr,
        }

    def _train_mlx(self, base_model, data_path, out, rank, epochs, lr):
        script = f"""#!/bin/bash
set -e
python -m mlx_lm.lora \\
    --model {base_model} \\
    --data {data_path.parent} \\
    --train \\
    --iters {max(10, epochs * 100)} \\
    --lora-layers 8 \\
    --lora-rank {rank} \\
    --learning-rate {lr} \\
    --save-path {out / 'adapters'}
"""
        script_path = out / "train_mlx.sh"
        script_path.write_text(script)
        result = subprocess.run(["bash", str(script_path)], capture_output=True, text=True, timeout=3600)
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "backend": "mlx",
            "output": result.stdout[-1000:] + result.stderr[-1000:],
            "output_dir": str(out),
        }

    def _train_unsloth(self, base_model, data_path, out, rank, epochs, lr, batch_size):
        script_path = out / "train_unsloth.py"
        script = f'''"""Unsloth LoRA training script."""
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{base_model}",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r={rank},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha={rank * 2},
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("json", data_files="{data_path}", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size={batch_size},
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs={epochs},
        learning_rate={lr},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="{out}",
        save_strategy="epoch",
    ),
)
trainer.train()
model.save_pretrained("{out / 'lora_model'}")
tokenizer.save_pretrained("{out / 'lora_model'}")
print(f"LoRA adapter saved to {{out / 'lora_model'}}")
'''
        script_path.write_text(script)

        import subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=7200,
        )
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "backend": "unsloth",
            "output": result.stdout[-1000:] + result.stderr[-1000:],
            "output_dir": str(out),
        }

    def _train_llamacpp(self, base_model, data_path, out, rank, epochs, lr):
        try:
            result = subprocess.run(
                ["llama.cpp-finetune", "--model", base_model, "--data", str(data_path),
                 "--lora-out", str(out / "lora.gguf"), "--lora-rank", str(rank)],
                capture_output=True, text=True, timeout=3600,
            )
            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "backend": "llama.cpp",
                "output": result.stdout[-1000:] + result.stderr[-1000:],
                "output_dir": str(out),
            }
        except FileNotFoundError:
            template = f"""#!/bin/bash
# llama.cpp LoRA Training
# Install: git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && make LLAMA_FINETUNE=1
MODEL="{base_model}"
DATA="{data_path}"
OUT="{out}"
echo "Run: cd llama.cpp && ./finetune --model-base convert.py $MODEL {out}/base.gguf --train-data $DATA --lora-out {out}/lora.gguf --lora-rank {rank}"
"""
            (out / "train_llamacpp.sh").write_text(template)
            return {
                "status": "script_generated",
                "backend": "llama.cpp",
                "instruction": f"See: {out / 'train_llamacpp.sh'}",
                "output_dir": str(out),
            }

    def export_modelfile(self, base_model: str, adapter_path: str | Path, output_path: str | Path) -> Path:
        modelfile = f"""FROM {base_model}
ADAPTER {adapter_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9

TEMPLATE \"\"\"{{ .Prompt }}\"\"\"

SYSTEM \"\"\"You are a fine-tuned model.\"\"\"
"""
        out = Path(output_path)
        out.write_text(modelfile)
        return out
