"""
Bangla Distill 3B — Curated High-Density Bengali Dataset Builder
==============================================================
Builds a clean, high-density Bengali training corpus for Qwen2.5-3B:
1. Extracts the 5,985 real encyclopedic articles from dual teacher corpora
   (Gemma-2-9B-IT + Qwen-3.5-9B), strictly eliminating the synthetic boilerplate rows.
2. Formats into ChatML format with rich, varied instructions.
3. Splits into 90% train / 10% validation.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILES = [
    os.path.join(SCRIPT_DIR, "teacher_corpus_gemma.jsonl"),
    os.path.join(SCRIPT_DIR, "teacher_corpus_qwen.jsonl"),
]
OUTPUT_TRAIN = os.path.join(SCRIPT_DIR, "train_chat_3b.jsonl")
OUTPUT_VAL = os.path.join(SCRIPT_DIR, "val_chat_3b.jsonl")
VAL_RATIO = 0.10
SEED = 42

SYSTEM_PROMPT = (
    "আপনি একজন অত্যন্ত দক্ষ ও তথ্যবহুল বাংলা এআই সহকারী। "
    "ব্যবহারকারীর প্রশ্নের উত্তর সঠিক ঐতিহাসিক ও বস্তুনিষ্ঠ তথ্যসহ সাবলীল বাংলায় বিস্তারিতভাবে দিন।"
)

QUESTION_TEMPLATES = [
    "{title} সম্পর্কে বিস্তারিত লিখুন।",
    "{title} সম্পর্কে বলুন।",
    "{title} কী? এর ঐতিহাসিক ও প্রাসঙ্গিক তথ্য বিস্তারিত ব্যাখ্যা করুন।",
    "{title} সম্পর্কে একটি তথ্যবহুল ও প্রাঞ্জল প্রবন্ধ লিখুন।",
    "{title} নিয়ে বিশদ আলোচনা করুন।",
    "{title} সম্পর্কে আমাকে বিস্তারিত জানান।",
    "{title} এর গুরুত্ব, ইতিহাস ও বিবরণ দিন।",
    "{title} — এই বিষয়ে বিস্তারিত তথ্য দিন।",
    "আমাকে {title} সম্পর্কে বিস্তারিত তথ্য শেখান।",
    "{title} কী এবং এটি কেন তাৎপর্যপূর্ণ?",
]

def make_question(title: str, deterministic_seed: str) -> str:
    idx = int(hashlib.md5(deterministic_seed.encode("utf-8")).hexdigest(), 16) % len(QUESTION_TEMPLATES)
    return QUESTION_TEMPLATES[idx].format(title=title)

def main():
    print("=" * 65)
    print("🇧🇩 Bangla Distill 3B — Curating High-Density Dataset")
    print("=" * 65)

    all_records = []
    skipped_boilerplate = 0

    for fpath in CORPUS_FILES:
        fname = os.path.basename(fpath)
        if not os.path.exists(fpath):
            print(f"[WARN] File not found: {fpath}")
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
                rec_id = raw.get("id", count)

                # Filter out synthetic boilerplate titles containing '—'
                if "—" in title:
                    skipped_boilerplate += 1
                    continue

                # Filter out short or empty texts
                if len(text) < 60 or not title:
                    continue

                seed_str = f"{teacher}_{rec_id}_{title}"
                q = make_question(title, seed_str)

                all_records.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": text.strip()}
                    ]
                })
                count += 1

        print(f"  [{fname}] Extracted {count} genuine knowledge articles")

    print(f"\nFiltered out {skipped_boilerplate} synthetic boilerplate rows.")
    print(f"Total high-quality genuine records: {len(all_records)}")

    random.seed(SEED)
    random.shuffle(all_records)

    val_count = max(100, int(len(all_records) * VAL_RATIO))
    val_set = all_records[:val_count]
    train_set = all_records[val_count:]

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        for r in val_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Train set saved: {OUTPUT_TRAIN} ({len(train_set)} records, {os.path.getsize(OUTPUT_TRAIN)/(1024**2):.2f} MB)")
    print(f"✅ Val set saved  : {OUTPUT_VAL} ({len(val_set)} records, {os.path.getsize(OUTPUT_VAL)/(1024**2):.2f} MB)")

if __name__ == "__main__":
    main()
