"""
Bangla Distill — Unsloth Fast Training & DPO Pipeline (Kaggle / Linux Script)
=============================================================================
Fine-tunes Qwen2.5-0.5B-Instruct on Bengali ChatML dataset with Unsloth fast kernels
and performs DPO reinforcement alignment.

Usage on Kaggle:
    python train_unsloth_kaggle.py
"""

import os
import sys
import shutil
import urllib.request
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

# 1. Environment & Paths
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "./kaggle_working"
os.makedirs(WORKING_DIR, exist_ok=True)

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = os.path.join(WORKING_DIR, "bangla-student-unsloth")

# 2. Locate or Download Datasets
TRAIN_FILE = None
VAL_FILE = None
search_roots = ["/kaggle/input", WORKING_DIR, "."]
for root_dir in search_roots:
    if os.path.exists(root_dir):
        for root, dirs, files in os.walk(root_dir):
            if "train_chat.jsonl" in files and not TRAIN_FILE:
                TRAIN_FILE = os.path.join(root, "train_chat.jsonl")
            if "val_chat.jsonl" in files and not VAL_FILE:
                VAL_FILE = os.path.join(root, "val_chat.jsonl")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/AshmitNur/Distilation/main"
if not TRAIN_FILE or not os.path.exists(TRAIN_FILE):
    print("Auto-downloading train_chat.jsonl from GitHub...")
    TRAIN_FILE = os.path.join(WORKING_DIR, "train_chat.jsonl")
    urllib.request.urlretrieve(f"{GITHUB_RAW_BASE}/train_chat.jsonl", TRAIN_FILE)

if not VAL_FILE or not os.path.exists(VAL_FILE):
    print("Auto-downloading val_chat.jsonl from GitHub...")
    VAL_FILE = os.path.join(WORKING_DIR, "val_chat.jsonl")
    urllib.request.urlretrieve(f"{GITHUB_RAW_BASE}/val_chat.jsonl", VAL_FILE)

print(f"✅ Training Dataset: {TRAIN_FILE}")
print(f"✅ Validation Dataset: {VAL_FILE}")

# 3. Load Model with Unsloth
print(f"\n[1/4] Loading {BASE_MODEL_NAME} with Unsloth FastLanguageModel...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 4. Format Dataset
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
raw_dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

def format_prompts(batch):
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False) for c in batch["messages"]]}

dataset = raw_dataset.map(format_prompts, batched=True)

# 5. SFT Training Stage
print("\n[2/4] Starting SFT Distillation with Assistant-Only Loss Masking...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=20,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer.train()

# 6. DPO Preference Reinforcement Stage
print("\n[3/4] Starting DPO Preference Reinforcement Step...")
PatchDPOTrainer()

preference_pairs = [
    {
        "prompt": "বাংলাদেশের ইতিহাস ও মুক্তিযুদ্ধ সম্পর্কে বলুন।",
        "chosen": "১৯৭১ সালের মহান মুক্তিযুদ্ধ বাঙালি জাতির ইতিহাসে গৌরবময় অধ্যায়। দীর্ঘ নয় মাসের রক্তক্ষয়ী সংগ্রাম ও ৩০ লাখ শহীদের আত্মত্যাগের বিনিময়ে আমরা অর্জন করেছি স্বাধীন বাংলাদেশ।",
        "rejected": "বাংলাদেশের ইতিহাস একটি গোষ্ঠী যা ১৯৮৩ সালে জন্মগ্রহণ করেছিল। আলফেলুল স্টিফোডিড শহীদ করেন accopted accopted."
    },
    {
        "prompt": "কৃত্রিম বুদ্ধিমত্তা (AI) কী?",
        "chosen": "কৃত্রিম বুদ্ধিমত্তা (Artificial Intelligence) হলো কম্পিউটার বিজ্ঞানের এমন একটি শাখা, যার মাধ্যমে কম্পিউটার বা মেশিন মানুষের বুদ্ধিমত্তা, যুক্তি এবং শেখার ক্ষমতা অনুকরণ করে।",
        "rejected": "কৃত্রিম বুদ্ধিমত্তা হলো যেমন সোলালেজন হিটান ছিলেন, আমি ভাষণ প্রশ্ন দেয় ipmap mite mite."
    }
]
dpo_ds = Dataset.from_list(preference_pairs)
dpo_args = DPOConfig(
    output_dir=os.path.join(WORKING_DIR, "dpo_out"),
    beta=0.1,
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    max_length=512,
    max_prompt_length=256,
    report_to=[],
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    train_dataset=dpo_ds,
    tokenizer=tokenizer,
)
dpo_trainer.train()

# 7. Save and Package
print("\n[4/4] Packaging Final LoRA Adapter Weights...")
FINAL_ADAPTER = os.path.join(WORKING_DIR, "final-lora-adapter-unsloth")
model.save_pretrained(FINAL_ADAPTER)
tokenizer.save_pretrained(FINAL_ADAPTER)

zip_file = shutil.make_archive(os.path.join(WORKING_DIR, "bangla_distill_unsloth_adapter"), "zip", FINAL_ADAPTER)
print(f"\n🎉 All Pipeline Steps Complete! Download zip from:\n   {zip_file}")
