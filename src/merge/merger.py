"""Model merging - merge fine-tuned LoRA adapters or model weights.

Supports:
- Linear interpolation (task arithmetic)
- SLERP (spherical linear interpolation)
- TIES-Merging (trim, elect, sign)
- LoRA adapter stacking
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional


class ModelMerger:
    def __init__(self, output_dir: str | Path = "data/merged") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_lora_adapters(self, base_model: str, adapter_paths: list[str | Path], method: str = "stack") -> dict:
        """Merge multiple LoRA adapters into a single Ollama Modelfile."""
        out = self.output_dir / "merged_lora"
        out.mkdir(parents=True, exist_ok=True)

        if method == "stack":
            modelfile_parts = [f"FROM {base_model}"]
            for i, adapter in enumerate(adapter_paths):
                modelfile_parts.append(f"ADAPTER {adapter}")
            modelfile = "\n".join(modelfile_parts)
            path = out / "Modelfile"
            path.write_text(modelfile)
            return {
                "status": "created",
                "method": "stack",
                "modelfile": str(path),
                "instruction": f"ollama create merged-model -f {path}",
            }

        return {"status": "error", "message": f"Unknown method: {method}"}

    def merge_gguf(self, model_paths: list[str | Path], method: str = "slerp", weights: Optional[list[float]] = None) -> dict:
        """Merge GGUF models using external tools (llama.cpp or mergekit)."""
        out = self.output_dir / "merged_gguf"
        out.mkdir(parents=True, exist_ok=True)

        if len(model_paths) < 2:
            return {"status": "error", "message": "Need at least 2 models to merge"}

        if weights is None:
            weights = [1.0 / len(model_paths)] * len(model_paths)

        if method == "slerp":
            script = out / "merge_slerp.py"
            script_content = f'''"""SLERP merge of GGUF models."""
import numpy as np
import struct
from pathlib import Path

models = {[str(p) for p in model_paths]}
weights = {weights}
output_path = "{out / 'merged.gguf'}"

print(f"Merging {{len(models)}} models via SLERP")
print(f"Weights: {{weights}}")
print(f"Output: {{output_path}}")

# This is a scaffold - actual GGUF merging requires llama.cpp's merge tool
# or mergekit. Install via: pip install mergekit
print("\\nTo perform actual merge:")
print(f"  mergekit-slerp {{' '.join(models)}} --weight {{weights[0]}} -o {{output_path}}")
print("Or use llama.cpp's merge tool:")
print(f"  llama-merge --base {{models[0]}} --target {{models[1]}} -o {{output_path}}")
'''
            script.write_text(script_content)
            return {
                "status": "script_generated",
                "method": "slerp",
                "script": str(script),
                "instruction": f"python {script}",
                "models": [str(p) for p in model_paths],
                "weights": weights,
            }
        elif method == "ties":
            script = out / "merge_ties.py"
            script_content = f'''"""TIES-Merging of models (Trim, Elect, Sign)."""
models = {[str(p) for p in model_paths]}
weights = {weights}
output_path = "{out / 'merged_ties.gguf'}"

print("TIES-Merging requires mergekit. Install via: pip install mergekit")
print(f"\\nmergekit-ties {{' '.join(models)}} --weights {{' '.join(map(str, weights))}} -o {{output_path}}")
'''
            script.write_text(script_content)
            return {
                "status": "script_generated",
                "method": "ties",
                "script": str(script),
                "instruction": f"python {script}",
            }

        return {"status": "error", "message": f"Unknown method: {method}"}

    def merge_task_arithmetic(self, base_weights: str, task_vectors: list[tuple[str, float]], output_name: str = "merged") -> dict:
        """Task arithmetic: base + sum(sign * weight * task_vector)."""
        out = self.output_dir / output_name
        out.mkdir(parents=True, exist_ok=True)

        task_list = "\n".join(f"  - {vec} (weight {w})" for vec, w in task_vectors)
        script = out / "merge_task_arithmetic.py"
        script_content = f'''"""Task arithmetic model merging."""
base = "{base_weights}"
task_vectors = {[(v, w) for v, w in task_vectors]}
output = "{out / 'merged.gguf'}"

print(f"Base model: {{base}}")
print("Task vectors:")
for vec, weight in task_vectors:
    print(f"  - {{vec}} (weight {{weight}})")
print(f"\\nTask arithmetic requires mergekit or a PyTorch implementation.")
print("Install: pip install mergekit")
print(f"\\nmergekit-arithmetic {{base}} {' '.join(v[0] for v in task_vectors)} --weights {' '.join(str(v[1]) for v in task_vectors)} -o {{output}}")
'''
        script.write_text(script_content)
        return {
            "status": "script_generated",
            "method": "task_arithmetic",
            "script": str(script),
            "output_dir": str(out),
        }

    def merge_pytorch_checkpoints(self, checkpoint_paths: list[str | Path], weights: Optional[list[float]] = None, output_path: str | Path = "checkpoints/merged.pt") -> dict:
        """Merge PyTorch model checkpoints directly via weight averaging."""
        import torch

        if len(checkpoint_paths) < 2:
            return {"status": "error", "message": "Need at least 2 checkpoints"}

        if weights is None:
            weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)

        if abs(sum(weights) - 1.0) > 0.01:
            weights = [w / sum(weights) for w in weights]

        checkpoints = []
        config = None
        for cp_path in checkpoint_paths:
            cp = torch.load(str(cp_path), map_location="cpu")
            checkpoints.append(cp["model_state_dict"])
            if config is None:
                config = cp.get("model_config", {})

        merged_state = {}
        for key in checkpoints[0]:
            merged_tensor = None
            for i, cp in enumerate(checkpoints):
                if key in cp:
                    weighted = cp[key] * weights[i]
                    if merged_tensor is None:
                        merged_tensor = weighted
                    else:
                        merged_tensor += weighted
            if merged_tensor is not None:
                merged_state[key] = merged_tensor

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": merged_state,
            "model_config": config,
            "vocab_size": checkpoints[0].get("vocab_size", 0),
            "merge_method": "weighted_average",
            "merge_weights": weights,
        }, str(out))

        return {
            "status": "completed",
            "method": "weighted_average",
            "checkpoints": [str(p) for p in checkpoint_paths],
            "weights": weights,
            "output": str(out),
        }
