"""
Bangla Distill v2 — Corpus Reformatter
=======================================
Converts the raw-text teacher corpus (teacher_corpus_gemma.jsonl + teacher_corpus_qwen.jsonl)
into ChatML instruction-response pairs suitable for TRL SFTTrainer with `messages` format.

Usage:
    python prepare_chat_dataset.py

Outputs:
    train_chat.jsonl  — 90% training split
    val_chat.jsonl    — 10% validation split
"""

import os
import sys
import json
import random
import hashlib

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILES = [
    os.path.join(SCRIPT_DIR, "teacher_corpus_gemma.jsonl"),
    os.path.join(SCRIPT_DIR, "teacher_corpus_qwen.jsonl"),
]
OUTPUT_TRAIN = os.path.join(SCRIPT_DIR, "train_chat.jsonl")
OUTPUT_VAL = os.path.join(SCRIPT_DIR, "val_chat.jsonl")
VAL_RATIO = 0.10
SEED = 42

# --- Bengali System Prompt ---
# This tells the model its role: "You are a skilled Bengali language assistant.
# Answer questions thoroughly in Bengali."
SYSTEM_PROMPT = (
    "আপনি একজন বাংলা ভাষায় দক্ষ সহকারী। "
    "ব্যবহারকারীর প্রশ্নের উত্তর সঠিক তথ্যসহ বিস্তারিতভাবে বাংলায় দিন।"
)

# --- Diverse Bengali Question Templates ---
# Each template takes a {title} placeholder. Using diverse phrasing prevents
# the model from overfitting on a single question pattern.
QUESTION_TEMPLATES = [
    "{title} সম্পর্কে বিস্তারিত লিখুন।",
    "{title} সম্পর্কে বলুন।",
    "{title} কী? বিস্তারিত ব্যাখ্যা করুন।",
    "{title} সম্পর্কে একটি তথ্যবহুল প্রবন্ধ লিখুন।",
    "{title} নিয়ে আলোচনা করুন।",
    "{title} সম্পর্কে আমাকে জানান।",
    "{title} এর গুরুত্ব ও বিবরণ দিন।",
    "{title} — এই বিষয়ে বিস্তারিত তথ্য দিন।",
    "আমাকে {title} সম্পর্কে শেখান।",
    "{title} কী এবং এটি কেন গুরুত্বপূর্ণ?",
]


def make_question(title: str, deterministic_seed: str) -> str:
    """Pick a question template deterministically based on a hash of the seed string."""
    # Use a hash so the same (title, teacher) always maps to the same template,
    # ensuring reproducibility without randomness.
    idx = int(hashlib.md5(deterministic_seed.encode("utf-8")).hexdigest(), 16) % len(QUESTION_TEMPLATES)
    return QUESTION_TEMPLATES[idx].format(title=title)


def build_chat_record(title: str, text: str, teacher: str, record_id: int) -> dict:
    """Convert a single raw corpus entry into a ChatML messages record."""
    seed_str = f"{teacher}_{record_id}_{title}"
    user_question = make_question(title, seed_str)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": text.strip()},
        ]
    }


def main():
    print("=" * 60)
    print("Bangla Distill v2 — Corpus Reformatter")
    print("=" * 60)

    all_records = []

    for fpath in CORPUS_FILES:
        fname = os.path.basename(fpath)
        if not os.path.exists(fpath):
            print(f"[WARN] File not found, skipping: {fpath}")
            continue

        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)

                title = raw.get("title", "").strip()
                text = raw.get("text", "").strip()
                teacher = raw.get("teacher", "unknown")
                record_id = raw.get("id", count)

                # Skip entries with very short text (< 50 chars) or missing title
                if len(text) < 50 or not title:
                    continue

                chat_record = build_chat_record(title, text, teacher, record_id)
                all_records.append(chat_record)
                count += 1

        print(f"  [{fname}] Loaded {count} records")

    print(f"\nTotal records: {len(all_records)}")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(all_records)

    val_count = int(len(all_records) * VAL_RATIO)
    val_set = all_records[:val_count]
    train_set = all_records[val_count:]

    print(f"Train split: {len(train_set)} records")
    print(f"Val split:   {len(val_set)} records")

    # Write output files
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for record in train_set:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        for record in val_set:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nOutput files:")
    print(f"  {OUTPUT_TRAIN} ({os.path.getsize(OUTPUT_TRAIN) / 1024:.1f} KB)")
    print(f"  {OUTPUT_VAL} ({os.path.getsize(OUTPUT_VAL) / 1024:.1f} KB)")

    # Show a sample record
    print(f"\n--- Sample Record ---")
    sample = train_set[0]
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:150] + ("..." if len(msg["content"]) > 150 else "")
        print(f"  [{role}]: {content}")

    print("\nDone!")


if __name__ == "__main__":
    main()
