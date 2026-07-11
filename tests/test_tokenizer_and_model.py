import torch

from configs.model_config import MODEL_CONFIGS
from src.dataset import TextDataset
from src.model import RMSNorm, SwiGLU, TinyTransformer
from src.multimodal.planner import ContentPlanner
from src.multimodal.image import generate_image_artifact
from src.multimodal.video import generate_video_artifact
from src.multimodal.code import generate_code_artifact
from src.tokenizer import BPETokenizer


def test_bpe_tokenizer_round_trip():
    tokenizer = BPETokenizer(target_vocab_size=64)
    tokenizer.fit(["hello world", "hello there", "world hello"])

    tokens = tokenizer.encode("hello world")
    decoded = tokenizer.decode(tokens)

    assert decoded == "hello world"
    assert len(tokens) > 0
    assert tokenizer.vocab_size > 1


def test_text_dataset_build():
    tokenizer = BPETokenizer(target_vocab_size=64)
    training_texts = ["hello world", "hello there"] * 16
    tokenizer.fit(training_texts)
    dataset = TextDataset.from_texts(training_texts, tokenizer, block_size=8, stride=1)

    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (8,)
    assert y.shape == (8,)


def test_rms_norm_and_swiglu_shapes():
    x = torch.randn(2, 4, 8)
    norm = RMSNorm(8)
    normalized = norm(x)
    assert normalized.shape == x.shape

    swiglu = SwiGLU(8, 16)
    y = swiglu(x)
    assert y.shape == x.shape


def test_tiny_transformer_forward_shape():
    config = MODEL_CONFIGS["micro"]
    model = TinyTransformer(
        vocab_size=128,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        max_context_len=config["block_size"],
        ff_hidden_size=config["ff_hidden_size"],
    )
    x = torch.randint(0, 128, (2, 8))
    logits = model(x)

    assert logits.shape == (2, 8, 128)


def test_classify_request_detects_modalities():
    planner = ContentPlanner()
    assert planner.classify_request("Write a Python script to scrape a website") == "code"
    assert planner.classify_request("Create a futuristic poster for a launch event") == "image"
    assert planner.classify_request("Make an infographic about AI safety") == "infographic"
    assert planner.classify_request("Generate a short cinematic promo video") == "video"
    assert planner.classify_request("Create a podcast script") == "audio"
    assert planner.classify_request("What is the meaning of life") == "text"


def test_generate_image_artifact_creates_png(tmp_path):
    import asyncio
    output_path = tmp_path / "hero.png"
    asyncio.run(generate_image_artifact("bright cyberpunk city", output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_video_artifact_creates_gif(tmp_path):
    import asyncio
    output_path = tmp_path / "promo.gif"
    asyncio.run(generate_video_artifact("launch trailer", output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_video_artifact_supports_storyboard_style(tmp_path):
    import asyncio
    output_path = tmp_path / "cinematic.mp4"
    asyncio.run(generate_video_artifact("cinematic promo with neon lights", output_path, scene_count=2, fps=8, seconds_per_scene=1.0))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_code_artifact_creates_python_file(tmp_path):
    import asyncio
    output_path = tmp_path / "app.py"
    asyncio.run(generate_code_artifact("build a CLI tool", output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0
