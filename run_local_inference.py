import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final-lora-adapter-v2")
SYSTEM_PROMPT = "আপনি একজন বাংলা ভাষায় দক্ষ সহকারী। ব্যবহারকারীর প্রশ্নের উত্তর সঠিক তথ্যসহ বিস্তারিতভাবে বাংলায় দিন।"

def main():
    parser = argparse.ArgumentParser(description="Run Bangla Distill Student Model locally on GPU")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat mode")
    parser.add_argument("--prompt", type=str, default=None, help="Custom single prompt to test")
    args = parser.parse_args()

    print("=" * 70)
    print("🇧🇩 BANGLA DISTILL — LOCAL GPU INFERENCE ENGINE")
    print("=" * 70)

    # 1. Device Diagnostics
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Running on CPU (slow!).")
        device = "cpu"
        dtype = torch.float32
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ GPU Detected: {gpu_name} ({vram:.1f} GB VRAM)")
        device = "cuda"
        dtype = torch.float16

    # 2. Load Tokenizer
    print(f"\n[1/3] Loading Tokenizer from:\n      {ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    print("      Tokenizer loaded successfully.")

    # 3. Load Base Model
    print(f"\n[2/3] Loading Base Model ({BASE_MODEL}) in FP16 precision...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    print(f"      Base model loaded in {time.time() - t0:.2f}s.")

    # 4. Attach LoRA Adapter
    print(f"\n[3/3] Attaching LoRA Adapter from:\n      {ADAPTER_DIR}")
    t0 = time.time()
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    print(f"      LoRA Adapter attached in {time.time() - t0:.2f}s.")
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"      Total Parameters: {total_params:.1f}M")
    print(f"      Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB VRAM")

    def generate(prompt, max_new_tokens=250, temperature=0.3, top_p=0.8, repetition_penalty=1.2, use_lora=True):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            if isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            else:
                input_ids = inputs.to(device)
                attention_mask = None
        except Exception:
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(device)
            attention_mask = getattr(enc, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            if not use_lora:
                with model.disable_adapter():
                    out = model.generate(**gen_kwargs)
            else:
                out = model.generate(**gen_kwargs)

        new_tokens = out[0][input_ids.shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Custom single prompt
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("-" * 70)
        res = generate(args.prompt)
        print(f"Response:\n{res}")
        return

    # Interactive loop
    if args.interactive:
        print("\n" + "=" * 70)
        print("Interactive Chat Mode (Type 'quit' or 'exit' to end)")
        print("=" * 70)
        while True:
            try:
                user_prompt = input("\nYour Prompt (Bengali): ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_prompt or user_prompt.lower() in ["exit", "quit", "q"]:
                print("\nExiting interactive chat.")
                break
            print("\nGenerating...")
            t_gen = time.time()
            resp = generate(user_prompt)
            duration = time.time() - t_gen
            print(f"\nResponse ({duration:.2f}s):")
            print(resp)
            print("-" * 70)
        return

    # Benchmark test suite
    test_prompts = [
        {
            "category": "History & National Identity",
            "prompt": "বাংলাদেশের ইতিহাস ও স্বাধীনতা যুদ্ধ সম্পর্কে বলুন।",
            "english": "Tell me about the history and Liberation War of Bangladesh."
        },
        {
            "category": "Science & Artificial Intelligence",
            "prompt": "কৃত্রিম বুদ্ধিমত্তা (Artificial Intelligence) কী এবং এটি কীভাবে কাজ করে?",
            "english": "What is Artificial Intelligence and how does it work?"
        },
        {
            "category": "Geography & Nature",
            "prompt": "সুন্দরবনের গুরুত্ব এবং এর রয়েল বেঙ্গল টাইগার সম্পর্কে বিস্তারিত লিখুন।",
            "english": "Write in detail about the importance of Sundarbans and its Royal Bengal Tiger."
        },
        {
            "category": "Advice & Health",
            "prompt": "একজন শিক্ষার্থীর সুস্থ ও কর্মক্ষম থাকার জন্য তিনটি প্রয়োজনীয় নিয়ম কী?",
            "english": "What are three essential rules for a student to stay healthy and productive?"
        }
    ]

    print("\n" + "=" * 70)
    print("RUNNING BENCHMARK COMPARISON (BASE MODEL vs. DISTILLED STUDENT)")
    print("=" * 70)

    results_md = "# 🇧🇩 Bangla Distill — Local Evaluation Results\n\n"
    results_md += f"**Execution Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n"
    results_md += f"**Hardware:** {torch.cuda.get_device_name(0)} ({torch.cuda.memory_allocated() / (1024**2):.1f} MB Allocated)  \n"
    results_md += f"**Base Model:** {BASE_MODEL} (FP16)  \n"
    results_md += f"**Adapter:** LoRA rank=16, alpha=32  \n\n---\n\n"

    for i, item in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}] {item['category']}")
        print(f"Prompt: {item['prompt']}")
        print(f"English: {item['english']}")
        print("-" * 70)

        # 1. Base Model (No LoRA)
        print("Generating Base Model response (No LoRA)...")
        t_base = time.time()
        base_resp = generate(item["prompt"], use_lora=False)
        base_time = time.time() - t_base

        # 2. Distilled Student (With LoRA)
        print("Generating Distilled Student response (With LoRA)...")
        t_lora = time.time()
        student_resp = generate(item["prompt"], use_lora=True)
        student_time = time.time() - t_lora

        print(f"\n🔴 BASE MODEL ({base_time:.2f}s):\n{base_resp}\n")
        print(f"🟢 DISTILLED STUDENT ({student_time:.2f}s):\n{student_resp}\n")
        print("=" * 70)

        results_md += f"### Test {i}: {item['category']}\n\n"
        results_md += f"**Prompt:** `{item['prompt']}`  \n"
        results_md += f"*Translation:* {item['english']}  \n\n"
        results_md += "| Model Variant | Generation Time | Output Text |\n"
        results_md += "| :--- | :--- | :--- |\n"
        results_md += f"| **Vanilla Base (Qwen2.5-0.5B)** | {base_time:.2f}s | {base_resp.replace(chr(10), '<br>')} |\n"
        results_md += f"| **Distilled Student (Bangla LoRA)** | {student_time:.2f}s | {student_resp.replace(chr(10), '<br>')} |\n\n"
        results_md += "---\n\n"

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_test_results.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(results_md)

    print(f"\n🎉 All tests completed successfully!")
    print(f"📄 Full comparison report saved to: {out_file}")

if __name__ == "__main__":
    main()
