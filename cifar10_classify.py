#!/usr/bin/env python3
"""
CIFAR-10 classification using ai.sooners.us (OpenAI-compatible Chat Completions).

Runs 5 different system prompts:
  • prompt_01: very vague ("what is this picture?")
  • prompt_02: strict format (single lowercase label from the 10 classes)
  • prompt_03: role-based "expert" classifier
  • prompt_04: guidance with short class definitions / decision hints
  • prompt_05: structured JSON output: {"label":"<class>"}

For each prompt:
  • Sample 100 images (10/class) from CIFAR-10 (fixed seed for reproducibility)
  • Send each as base64 data URL to /api/chat/completions with gemma3:4b
  • Collect predictions, compute accuracy, save confusion matrix and CSV
  • Log both "invalid" (unparseable) and "wrong" predictions to JSONL

Requires:
  pip install requests python-dotenv torch torchvision pillow scikit-learn matplotlib
Secrets (not committed):
  ~/.soonerai.env with:
    SOONERAI_API_KEY=...
    SOONERAI_BASE_URL=https://ai.sooners.us
    SOONERAI_MODEL=gemma3:4b
"""

import os
import io
import time
import base64
import random
import json
from typing import List, Dict, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ── Load secrets ──────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.expanduser("~"), ".soonerai.env"))
API_KEY = os.getenv("SOONERAI_API_KEY")
BASE_URL = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
MODEL = os.getenv("SOONERAI_MODEL", "gemma3:4b")

if not API_KEY:
    raise RuntimeError("Missing SOONERAI_API_KEY in ~/.soonerai.env")

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 1337
SAMPLES_PER_CLASS = 10
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# TODO: Try different SYSTEM_PROMPT to improve accuracy.
# Five prompt variants (system messages)
SYSTEM_PROMPTS: Dict[str, str] = {
    # 01) Very vague (intentionally weak)
    "prompt_01": "What is this picture?",

    # 02) Very accurate requirements (strict, one label only)
    "prompt_02": (
        "You are a CIFAR-10 image classifier. "
        "Return exactly one label (lowercase) from: "
        "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
        "Do not add punctuation, emojis, or extra words."
    ),

    # 03) Role-based expert classifier (still strict)
    "prompt_03": (
        "You are a computer vision expert trained on CIFAR-10. "
        "Classify the image into exactly one lowercase label from the 10 categories. "
        "Respond with the label only."
    ),

    # 04) Decision hints to reduce common confusions
    "prompt_04": (
        "Be precise and choose exactly one lowercase label. "
        "Hints: airplane (sky aircraft), ship (watercraft), "
        "automobile (passenger car), truck (cargo/pickup), "
        "bird (beak/wings), cat vs dog (ears/snout), "
        "horse (mane/long legs), deer (antlers), frog (small amphibian). "
        "Answer with only the label."
    ),

    # 05) Structured (JSON label) — we still allow plain label via USER_INSTRUCTION constraint
    "prompt_05": (
        "Output only a single label, or a JSON object like {\"label\":\"dog\"}. "
        "No explanations."
    ),
}

# Constrain the model’s output to *one* of the valid labels.
USER_INSTRUCTION = f"""
Classify this CIFAR-10 image. Respond with exactly one label from this list:
{', '.join(CLASSES)}
Your reply must be just the label, nothing else.
""".strip()

# Simple synonym normalizer to coerce common outputs to CIFAR-10 labels
NORMALIZE = {
    "car": "automobile",
    "auto": "automobile",
    "vehicle": "automobile",
    "plane": "airplane",
    "aeroplane": "airplane",
    "boat": "ship",
    "vessel": "ship",
    "pickup": "truck",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encode a PIL image to base64 JPEG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def post_chat_completion_image(
    image_data_url: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """
    Send an image + instruction to /api/chat/completions and return the text reply.
    Uses OpenAI-style content parts with an image_url Data URL.
    """
    url = f"{base_url}/api/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def normalize_label(text: str) -> str:
    """
    Map model reply to a valid CIFAR-10 class if possible.
    Handles:
      - single word (e.g., "dog")
      - phrases (first token)
      - synonyms (e.g., "car" -> "automobile")
      - JSON objects like {"label":"dog"}
    """
    t = text.strip()

    # Try JSON first (prompt_05)
    if t.startswith("{") and t.endswith("}"):
        try:
            obj = json.loads(t)
            cand = str(obj.get("label", "")).strip().lower()
            if cand in NORMALIZE:
                cand = NORMALIZE[cand]
            if cand.endswith("s") and cand[:-1] in CLASSES:
                cand = cand[:-1]
            if cand in CLASSES:
                return cand
        except Exception:
            pass  # fall through

    t = t.lower()
    # Take first token if sentence
    tok = t.split()[0].strip(",.!?;:\"'()[]{}")
    tok = NORMALIZE.get(tok, tok)
    if tok.endswith("s") and tok[:-1] in CLASSES:
        tok = tok[:-1]
    if tok in CLASSES:
        return tok

    # Loose fallback: if any class name appears as substring
    for c in CLASSES:
        if c in t:
            return c

    return "__unknown__"

# ── Data: stratified sample of 100 images (10/class) ─────────────────────────
def stratified_sample_cifar10(root: str = "./data") -> List[Tuple[Image.Image, int, int]]:
    """
    Download CIFAR-10 (train split) and return a list of (PIL_image, target, idx):
    exactly SAMPLES_PER_CLASS per class (fixed seed for reproducibility).
    """
    ds = CIFAR10(root=root, train=True, download=True)
    per_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):
        per_class[label].append(idx)

    random.seed(SEED)
    selected = []
    for label in range(10):
        chosen = random.sample(per_class[label], SAMPLES_PER_CLASS)
        for idx in chosen:
            img, tgt = ds[idx]
            selected.append((img, tgt, idx))
    return selected

def evaluate_and_plot(y_true: List[int], y_pred: List[int], out_png: str) -> float:
    """
    Compute accuracy (invalid predictions count as wrong).
    Plot/save a 10x10 confusion matrix using only valid predictions.
    """
    correct = sum(1 for t, p in zip(y_true, y_pred) if p == t)
    acc = correct / len(y_true)

    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p in range(10)]
    if valid_pairs:
        y_t = [t for (t, _) in valid_pairs]
        y_p = [p for (_, p) in valid_pairs]
        cm = confusion_matrix(y_t, y_p, labels=list(range(10)))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=45, colorbar=False)
        plt.title(f"CIFAR-10 Confusion Matrix (gemma3:4b) | Acc: {acc:.3f}")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
    else:
        print("[warn] No valid predictions to plot confusion matrix.")

    return acc

def run_prompt(
    prompt_id: str,
    system_prompt: str,
    samples: List[Tuple[Image.Image, int, int]],
    sleep_sec: float = 0.2,
    max_retries: int = 3,
) -> Dict:
    """
    Classify all sampled images with the given system prompt.
    Returns dict with rows, y_true, y_pred, bad (invalid+wrong), acc, csv, cm.
    """
    rows = []
    y_true: List[int] = []
    y_pred: List[int] = []
    bad: List[Dict] = []

    print(f"\n[run] {prompt_id}")
    for i, (img, tgt, idx) in enumerate(samples, start=1):
        data_url = pil_to_base64_jpeg(img)

        raw_reply = ""
        pred_label = "__unknown__"
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_reply = post_chat_completion_image(
                    image_data_url=data_url,
                    system_prompt=system_prompt,
                    model=MODEL,
                    base_url=BASE_URL,
                    api_key=API_KEY,
                    temperature=0.0,
                )
                pred_label = normalize_label(raw_reply)
                break
            except Exception as e:
                last_err = e
                time.sleep(0.4 * attempt)

        if pred_label in CLASSES:
            pred_idx = CLASSES.index(pred_label)
        else:
            pred_idx = -1

        y_true.append(tgt)
        y_pred.append(pred_idx)

        true_label = CLASSES[tgt]
        shown_pred = pred_label if pred_idx != -1 else "__error__"
        shown_raw = raw_reply if raw_reply else f"[error] {last_err}"
        print(f"[{i:03d}/100] true={true_label:>10s} | pred={shown_pred:>10s} | raw='{shown_raw}'")

        rows.append({
            "i": i,
            "idx": idx,
            "true": true_label,
            "pred": pred_label if pred_idx != -1 else "__error__",
            "raw": shown_raw,
            "prompt_id": prompt_id,
        })

        # Log all non-perfect cases: invalid OR wrong
        if pred_idx == -1 or pred_idx != tgt:
            bad.append({
                "i": i,
                "idx": idx,
                "true": true_label,
                "pred": pred_label if pred_idx != -1 else "__error__",
                "raw_reply": shown_raw,
                "prompt_id": prompt_id,
                "kind": "invalid" if pred_idx == -1 else "wrong",
            })

        time.sleep(sleep_sec)  # be gentle to the API

    cm_path = f"{prompt_id}_confusion_matrix.png"
    acc = evaluate_and_plot(y_true, y_pred, out_png=cm_path)

    # Save CSV
    import csv
    csv_path = f"predictions_{prompt_id}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["i", "idx", "true", "pred", "raw", "prompt_id"])
        w.writeheader()
        w.writerows(rows)

    # Save misclassifications (JSONL)
    with open(f"misclassifications_{prompt_id}.jsonl", "w") as f:
        for row in bad:
            f.write(json.dumps(row) + "\n")

    print(f"[result] {prompt_id} acc={acc:.3f} | saved {csv_path}, {cm_path}, misclassifications_{prompt_id}.jsonl")
    return {
        "rows": rows,
        "y_true": y_true,
        "y_pred": y_pred,
        "bad": bad,
        "acc": acc,
        "csv": csv_path,
        "cm": cm_path,
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Preparing CIFAR-10 sample (100 images, fixed seed)...")
    samples = stratified_sample_cifar10()

    results_by_prompt: Dict[str, Dict] = {}
    acc_by_prompt: Dict[str, float] = {}

    for pid, sys_prompt in SYSTEM_PROMPTS.items():
        res = run_prompt(pid, sys_prompt, samples)
        results_by_prompt[pid] = res
        acc_by_prompt[pid] = res["acc"]

    # Summary
    best_pid = max(acc_by_prompt, key=acc_by_prompt.get)
    print("\n=== Summary ===")
    for pid, acc in acc_by_prompt.items():
        print(f"{pid}: accuracy = {acc:.3f}")
    print(f"Best prompt: {best_pid} → see {results_by_prompt[best_pid]['cm']} and {results_by_prompt[best_pid]['csv']}")

    meta = {
        "seed": SEED,
        "model": MODEL,
        "base_url": BASE_URL,
        "samples_per_class": SAMPLES_PER_CLASS,
        "accuracy_by_prompt": acc_by_prompt,
        "best_prompt": best_pid,
        "classes": CLASSES,
    }
    with open("run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved run_meta.json")


if __name__ == "__main__":
    main()