# Winner Model — Hardware Requirements

## Current System
- CPU: Intel i5-9600K
- RAM: 24 GB
- GPU: None
- This can run text-based AI (chat via Ollama) but NOT video, image, or advanced generation.

---

## Goal: Fully Independent AI Machine

We want to run everything locally — no subscriptions, no API fees:
- Large Language Models (Llama 3, Qwen, Mistral)
- AI Video Generation (Stable Video Diffusion, etc.)
- AI Image Generation (Stable Diffusion, FLUX)
- Audio / Music Generation
- Custom Model Training & Fine-Tuning

The GPU is the only critical component. Everything else is secondary.

---

## Option 1: Best Value (Recommended)

**Total: ~$1,000 – $1,300**

| Component | Item | Price |
|-----------|------|-------|
| GPU | Used RTX 3090 (24GB VRAM) | $700 – $900 |
| RAM | Upgrade to 64GB DDR4 | $80 – $120 |
| PSU | 850W power supply | $100 – $130 |
| Storage | 2TB NVMe SSD | $100 – $140 |

**Capabilities:**
- Llama 3 70B (quantized) ✓
- Qwen 32B, Mistral 22B ✓
- FLUX / Pro image generation ✓
- AI video generation (SVD, etc.) ✓
- Model fine-tuning ✓
- Everything we need

## Option 2: Budget Starter

**Total: ~$260 – $330**

| Component | Item | Price |
|-----------|------|-------|
| GPU | Used RTX 3060 (12GB VRAM) | $200 – $250 |
| PSU | 650W power supply | $60 – $80 |

**Capabilities:**
- 7B-13B LLMs ✓
- Stable Diffusion XL images ✓
- Basic video models (limited) ✓
- Larger models and advanced video ❌

## Option 3: No-Compromise / Future-Proof

**Total: ~$3,000 – $3,600**

| Component | Item | Price |
|-----------|------|-------|
| GPU | RTX 5090 (32GB VRAM) | $2,000 – $2,200 |
| CPU | AMD Ryzen 9 or Intel i9 | $400 – $600 |
| RAM | 64GB DDR5 | $150 – $200 |
| Motherboard | Z790 / X670E | $200 – $300 |
| PSU | 1000W | $150 – $200 |
| Storage | 2TB NVMe SSD | $100 – $140 |

**Capabilities:**
- Everything runs faster and supports larger models
- 70B models at higher precision
- Faster video/image generation
- Multi-model experimentation

---

## What Each Component Does

| Component | Why It Matters |
|-----------|---------------|
| **GPU** | The ONLY thing that matters for AI. More VRAM = bigger/faster models. |
| **RAM** | 64GB lets you load large models and datasets. 24GB (current) is a bottleneck. |
| **PSU** | RTX 3090 needs 850W minimum. Don't skip this. |
| **Storage** | Models take 10-100GB each. 2TB gives room to grow. |

## Key Insight

**A used RTX 3090 is the best value in AI hardware.** It has the same 24GB VRAM as a $2,000 RTX 4090 but costs half as much. Most AI workloads are VRAM-limited, not speed-limited.

---

## Where to Buy (2026)

| Item | Where |
|------|-------|
| Used GPU | eBay, r/hardwareswap (Reddit), Facebook Marketplace |
| New GPU | Amazon, Newegg, B&H Photo |
| Other parts | Amazon, Micro Center (if local) |

## After Purchase

When you have the hardware, the system is already built and ready. We just need to:
1. Install the GPU and drivers
2. Enable GPU acceleration in the code
3. Replace Ollama inference with direct GPU models
4. Enable local video/image generation pipelines

Everything is already coded and waiting.
