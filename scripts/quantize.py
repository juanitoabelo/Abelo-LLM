#!/usr/bin/env python3
"""
Quantization pipeline for the custom LLM system.

Converts models to GGUF format and applies quantization levels:
  - q4_0: 4-bit, symmetric, no blocks (fastest)
  - q4_K_M: 4-bit K-quant, medium size (balanced)
  - q5_K_M: 5-bit K-quant, medium (higher quality)
  - q8_0: 8-bit (highest quality, largest)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


QUANT_TYPES = ["q4_0", "q4_K_M", "q5_K_M", "q8_0"]

# Approximate size ratios vs fp16
QUANT_SIZES = {
    "q4_0": 0.27,
    "q4_K_M": 0.29,
    "q5_K_M": 0.35,
    "q8_0": 0.53,
}


def find_llamacpp_tools() -> tuple[Optional[Path], Optional[Path]]:
    convert = shutil.which("convert.py") or shutil.which("convert")
    quantize = shutil.which("quantize") or shutil.which("quantize.exe")
    if not quantize:
        for p in Path("/usr/local/bin").glob("*quantize*"):
            quantize = p
            break
    for p in Path.home().glob("llama.cpp/**/quantize"):
        quantize = p
        break
    for p in Path.home().glob("llama.cpp/**/convert.py"):
        convert = p
        break
    return (Path(convert) if convert else None, Path(quantize) if quantize else None)


def convert_pytorch_to_gguf(
    model_dir: str | Path,
    output_path: str | Path,
    convert_tool: Path,
) -> Path:
    out = Path(output_path)
    cmd = [
        sys.executable, str(convert_tool),
        str(model_dir),
        "--outfile", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def quantize_gguf(
    input_gguf: str | Path,
    output_path: str | Path,
    quant_type: str,
    quantize_tool: Path,
) -> Path:
    out = Path(output_path)
    cmd = [str(quantize_tool), str(input_gguf), str(out), quant_type]
    subprocess.run(cmd, check=True)
    return out


def create_ollama_modelfile(
    model_name: str,
    gguf_path: str | Path,
    output_dir: str | Path,
    template: str = "{{ .Prompt }}",
    system: str = "",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = out_dir / "Modelfile"
    content = f"""FROM {gguf_path}
TEMPLATE \"\"\"{template}\"\"\"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
    if system:
        content += f'SYSTEM """{system}"""\n'
    modelfile_path.write_text(content)
    return modelfile_path


def ollama_create(model_name: str, modelfile_path: str | Path) -> None:
    subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        check=True,
    )


def estimate_sizes(model_size_gb: float) -> dict[str, float]:
    return {
        qtype: round(model_size_gb * ratio, 2)
        for qtype, ratio in QUANT_SIZES.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize models via llama.cpp")
    sub = parser.add_subparsers(dest="command")

    quant_cmd = sub.add_parser("quantize", help="Quantize a model")
    quant_cmd.add_argument("input", help="Input GGUF file or model directory")
    quant_cmd.add_argument("--quant-type", choices=QUANT_TYPES, default="q4_K_M")
    quant_cmd.add_argument("--output-dir", default="data/quantized")
    quant_cmd.add_argument("--ollama-name", help="Register in Ollama after quantization")
    quant_cmd.add_argument("--convert-first", action="store_true", help="Convert PyTorch to GGUF first")
    quant_cmd.add_argument("--template", default="{{ .Prompt }}", help="Ollama template")
    quant_cmd.add_argument("--system", default="", help="Ollama system prompt")

    estimate_cmd = sub.add_parser("estimate", help="Estimate quantized sizes")
    estimate_cmd.add_argument("size_gb", type=float, help="Model fp16 size in GB")

    list_cmd = sub.add_parser("list", help="List available quant types")
    _ = sub.add_parser("detect", help="Detect llama.cpp tools")

    args = parser.parse_args()

    if args.command == "list":
        print("Available quantization types:")
        for qt in QUANT_TYPES:
            print(f"  {qt}")
        return

    if args.command == "detect":
        convert_tool, quantize_tool = find_llamacpp_tools()
        print(f"convert.py: {convert_tool or 'NOT FOUND'}")
        print(f"quantize:   {quantize_tool or 'NOT FOUND'}")
        return

    if args.command == "estimate":
        sizes = estimate_sizes(args.size_gb)
        print(f"Estimated sizes for {args.size_gb}GB fp16 model:")
        for qtype, size in sizes.items():
            print(f"  {qtype}: {size}GB")
        return

    if args.command == "quantize":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        convert_tool, quantize_tool = find_llamacpp_tools()

        input_path = Path(args.input)
        gguf_path: Optional[Path] = None

        if args.convert_first or input_path.is_dir():
            if not convert_tool:
                print("Error: convert.py not found. Use --detect to locate it.")
                sys.exit(1)
            gguf_path = output_dir / f"{input_path.stem}_fp16.gguf"
            print(f"Converting {input_path} -> {gguf_path}")
            gguf_path = convert_pytorch_to_gguf(input_path, gguf_path, convert_tool)
        else:
            gguf_path = input_path

        if not quantize_tool:
            print("Error: quantize tool not found. Use --detect to locate it.")
            sys.exit(1)

        out_name = f"{gguf_path.stem.replace('_fp16', '')}_{args.quant_type}.gguf"
        quant_path = output_dir / out_name
        print(f"Quantizing {gguf_path} -> {quant_path} ({args.quant_type})")
        quantize_gguf(gguf_path, quant_path, args.quant_type, quantize_tool)

        if args.ollama_name:
            print(f"Creating Ollama model '{args.ollama_name}'...")
            modelfile = create_ollama_modelfile(
                args.ollama_name, quant_path, output_dir,
                template=args.template, system=args.system,
            )
            ollama_create(args.ollama_name, modelfile)
            print(f"Ollama model '{args.ollama_name}' created")

        result = {
            "input": str(input_path),
            "output": str(quant_path),
            "quant_type": args.quant_type,
            "ollama_name": args.ollama_name,
        }
        (output_dir / "quant_result.json").write_text(json.dumps(result, indent=2))
        print(f"Done. Result saved to {output_dir / 'quant_result.json'}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
