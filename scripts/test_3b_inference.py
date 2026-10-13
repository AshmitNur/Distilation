import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "models", "bangla_distill_3b_adapter")

print("=" * 70)
print("🇧🇩 BANGLA DISTILL 3B — TEST EVALUATION ON RTX 4070")
print("=" * 70)
print(f"LoRA Adapter Path: {ADAPTER_DIR}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. Load Tokenizer
print("\n[1/3] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
print(f"      Tokenizer vocabulary: {len(tokenizer)} tokens")

# 2. Load Base Model in 4-bit NF4
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
print(f"\n[2/3] Loading {BASE_MODEL} in 4-bit NF4 precision...")
t0 = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print(f"      Base model loaded in {time.time() - t0:.2f}s!")
print(f"      VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

# 3. Attach 3B LoRA Adapter
print(f"\n[3/3] Attaching 3B LoRA Adapter...")
t0 = time.time()
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
print(f"      3B Adapter attached in {time.time() - t0:.2f}s!")
print(f"      Total VRAM with Adapter: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

# 4. Generate on Benchmark Prompt
test_prompts = [
    "বাংলাদেশের ইতিহাস ও স্বাধীনতা যুদ্ধ সম্পর্কে বলুন।",
    "কৃত্রিম বুদ্ধিমত্তা (Artificial Intelligence) কী এবং এটি কীভাবে কাজ করে?",
]

SYSTEM_PROMPT = "আপনি একজন বাংলা ভাষায় দক্ষ সহকারী। ব্যবহারকারীর প্রশ্নের উত্তর সঠিক তথ্যসহ বিস্তারিতভাবে বাংলায় দিন।"

for p in test_prompts:
    print("\n" + "=" * 70)
    print(f"❓ PROMPT: {p}")
    print("=" * 70)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": p}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda") if "attention_mask" in inputs else None
    
    t_start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=250,
            temperature=0.2,
            top_p=0.85,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.perf_counter() - t_start
    new_tokens = out[0][input_ids.shape[-1]:]
    tok_sec = len(new_tokens) / max(gen_time, 0.001)
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    print(f"⏱️ Generation Time: {gen_time:.2f}s | Speed: {tok_sec:.1f} tok/sec | Tokens: {len(new_tokens)}")
    print("\n🇧🇩 3B STUDENT OUTPUT:\n")
    print(response_text)
    print("-" * 70)
