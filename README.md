# Classification with gemma3:4b (VLM) via SoonerAI

This project uses the OpenAI-compatible Chat Completions API at `https://ai.sooners.us/api/chat/completions` with **gemma3:4b** to classify **100 CIFAR-10 images** (10 per class), experiment with **system prompts**, and plot **confusion matrices**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

