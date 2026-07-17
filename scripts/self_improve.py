"""Self-improvement loop: generate data → train → evaluate → improve.

Usage:
  python scripts/self_improve.py --seed "Explain quantum computing" --rounds 3
  python scripts/self_improve.py --dataset data/tiny_stories --rounds 5 --model-size small
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_improvement_round(round_num: int, seed_texts: list[str], model: str, output_dir: Path) -> dict:
    from src.training.distill import DistillationTrainer
    from src.training.data import DatasetBuilder
    from src.training.lora import LoRATrainer
    from src.llm.ollama import OllamaBackend
    from src.eval.harness import EvalHarness
    import asyncio

    round_dir = output_dir / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Self-Improvement Round {round_num}")
    print(f"{'='*50}")

    # 1. Generate synthetic data
    print("\n1. Generating synthetic training data...")
    trainer = DistillationTrainer(teacher_model=model, output_dir=str(round_dir))
    samples = trainer.generate_dataset(seed_texts, num_samples=50, temperature=0.8)
    data_path = trainer.save_jsonl(samples, f"round_{round_num}")
    print(f"   Generated {len(samples)} samples -> {data_path}")

    # 2. Build dataset
    print("\n2. Building training dataset...")
    builder = DatasetBuilder(output_dir=str(round_dir))
    converted = builder.convert_to_jsonl([{"text": s["text"]} for s in samples], f"train_round_{round_num}")
    print(f"   Dataset -> {converted}")

    # 3. Run LoRA fine-tuning (via Ollama Modelfile fallback)
    print(f"\n3. Fine-tuning {model} on generated data...")
    lora = LoRATrainer(backend="auto")
    result = lora.train(
        base_model=model,
        data_path=str(converted),
        output_dir=str(round_dir / "lora"),
        epochs=2,
    )
    print(f"   Fine-tune result: {result.get('status', 'unknown')}")
    if result.get("model_name"):
        print(f"   New model: {result['model_name']}")

    # 4. Evaluate
    print("\n4. Evaluating...")
    backend = OllamaBackend()

    async def _evaluate():
        harness = EvalHarness(model_fn=lambda p: asyncio.run(_get_response(backend, model, p)))
        test_questions = [
            {"question": s.get("prompt", s.get("text", ""))[:100], "answer": s.get("text", "")[:50]}
            for s in samples[:10]
        ]
        if test_questions:
            eval_result = harness.test_qa(test_questions)
            print(f"   Accuracy: {eval_result.get('accuracy', 0):.2%}")
            print(f"   Avg latency: {eval_result.get('avg_latency', 0):.2f}s")
            return eval_result
        return {}

    async def _get_response(backend, m, prompt):
        collected = ""
        async for chunk in backend.generate(prompt=prompt, model=m, max_tokens=50):
            collected += chunk
        return collected

    eval_result = asyncio.run(_evaluate())

    summary = {
        "round": round_num,
        "samples": len(samples),
        "eval": eval_result,
        "model": result.get("model_name", model),
    }

    (round_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n   Summary saved to {round_dir / 'summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-improvement loop for LLM")
    parser.add_argument("--seed", nargs="+", default=["Explain a complex topic simply", "Write a short story", "Solve a math problem step by step"], help="Seed texts to start generation")
    parser.add_argument("--rounds", type=int, default=3, help="Number of improvement rounds")
    parser.add_argument("--model", default="llama3.2:1b", help="Base model to improve")
    parser.add_argument("--dataset", default=None, help="Path to existing dataset file to include")
    parser.add_argument("--output", default="data/self_improve", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_texts = list(args.seed)

    if args.dataset:
        dataset_path = Path(args.dataset)
        if dataset_path.exists():
            with open(dataset_path) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and "text" in data:
                            seed_texts.append(data["text"][:200])
                    except json.JSONDecodeError:
                        pass
            print(f"Loaded {len(seed_texts)} seed texts from dataset")

    print(f"Starting self-improvement for {args.model}")
    print(f"Rounds: {args.rounds}, Seeds: {len(seed_texts)}")
    print(f"Output: {output_dir}")

    for r in range(1, args.rounds + 1):
        run_improvement_round(r, seed_texts, args.model, output_dir)

    print(f"\n{'='*50}")
    print("Self-improvement complete!")
    print(f"All results in: {output_dir}")


if __name__ == "__main__":
    main()
