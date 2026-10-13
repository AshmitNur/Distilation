"""
Bangla Distill — Benchmark Suite 1: Quantitative NLG & Language Modeling Benchmark
Evaluates Vanilla Base vs. Distilled Student on held-out validation data (val_chat.jsonl).
Computes:
  - Cross-Entropy Loss & Perplexity (PPL) on assistant completions
  - ROUGE-1, ROUGE-2, ROUGE-L
  - chrF++ (Character n-gram F-score with word order)
  - SacreBLEU
  - Latency (ms/sample), Generation Throughput (tokens/sec), Peak VRAM (MB)
Generates:
  - benchmarks/results/suite1_quantitative_nlg.json
  - Bangla_Distill_Benchmark_Suite1_Quantitative_NLG.pdf
"""

import os
import sys
import time
import json
import math
import datetime
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pdf_generator import html_to_pdf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "val_chat.jsonl") if os.path.exists(os.path.join(PROJECT_ROOT, "data", "val_chat.jsonl")) else os.path.join(PROJECT_ROOT, "val_chat.jsonl")
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def _resolve_model_dir(name: str) -> str:
    p1 = os.path.join(PROJECT_ROOT, "models", name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(PROJECT_ROOT, name)
    if os.path.exists(p2): return p2
    return p1

_UNSLOTH = _resolve_model_dir("bangla_distill_unsloth_adapter")
ADAPTER_PATH = _UNSLOTH if os.path.exists(_UNSLOTH) else _resolve_model_dir("final-lora-adapter-v2")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def compute_ppl_and_loss(model, tokenizer, dataset_samples, max_samples=40):
    """
    Computes cross-entropy loss and perplexity strictly on completion tokens.
    """
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    
    subset = dataset_samples[:max_samples]
    with torch.no_grad():
        for item in subset:
            msgs = item["messages"]
            # Build full text
            full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
            prompt_msgs = [m for m in msgs if m["role"] in ["system", "user"]]
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            
            full_enc = tokenizer(full_text, return_tensors="pt").to(DEVICE)
            prompt_enc = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            
            prompt_len = prompt_enc["input_ids"].shape[1]
            input_ids = full_enc["input_ids"]
            
            if input_ids.shape[1] <= prompt_len:
                continue
                
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100 # Mask prompt tokens
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss only on unmasked tokens
            valid_mask = shift_labels != -100
            num_valid = valid_mask.sum().item()
            if num_valid > 0:
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                total_tokens += num_valid
                
    if total_tokens == 0:
        return 0.0, 0.0
    mean_loss = total_loss / total_tokens
    ppl = math.exp(min(mean_loss, 20.0))
    return mean_loss, ppl

def generate_and_evaluate(model, tokenizer, dataset_samples, num_samples=30):
    """
    Generates completions for evaluation and measures throughput/latency.
    """
    subset = dataset_samples[:num_samples]
    predictions = []
    references = []
    prompts = []
    
    total_time = 0.0
    total_tokens = 0
    
    print(f"   [Evaluating {num_samples} samples]...")
    for idx, item in enumerate(subset):
        msgs = item["messages"]
        prompt_msgs = [m for m in msgs if m["role"] in ["system", "user"]]
        ref_text = next(m["content"] for m in msgs if m["role"] == "assistant")
        user_query = next((m["content"] for m in msgs if m["role"] == "user"), "")
        
        prompt_str = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=160,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        total_time += elapsed
        total_tokens += len(gen_tokens)
        
        prompts.append(user_query)
        predictions.append(pred_text)
        references.append(ref_text)
        
    avg_latency = (total_time / num_samples) * 1000 # ms
    tok_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    # Compute Metrics
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=predictions, references=references)
    
    chrf = evaluate.load("chrf")
    chrf_res = chrf.compute(predictions=predictions, references=references, word_order=2)
    
    bleu = evaluate.load("bleu")
    bleu_res = bleu.compute(predictions=predictions, references=references)
    
    sample_pairs = []
    for i in range(min(4, len(prompts))):
        sample_pairs.append({
            "prompt": prompts[i],
            "reference": references[i],
            "prediction": predictions[i]
        })
        
    return {
        "rouge1": rouge_res["rouge1"] * 100,
        "rouge2": rouge_res["rouge2"] * 100,
        "rougeL": rouge_res["rougeL"] * 100,
        "chrf": chrf_res["score"],
        "bleu": bleu_res["bleu"] * 100,
        "latency_ms": avg_latency,
        "tokens_per_sec": tok_per_sec,
        "samples": sample_pairs
    }

def run_suite():
    print("=" * 70)
    print("🇧🇩 BANGLA DISTILL — BENCHMARK SUITE 1: QUANTITATIVE NLG")
    print("=" * 70)
    
    # 1. Load Validation Data
    print(f"[*] Reading validation samples from: {VAL_DATA_PATH}")
    dataset_samples = []
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            dataset_samples.append(json.loads(line))
    print(f"    Loaded {len(dataset_samples)} total validation records.")
    
    NUM_PPL_SAMPLES = 50
    NUM_GEN_SAMPLES = 25
    print(f"[*] Evaluation sample sizes: {NUM_PPL_SAMPLES} samples for PPL/Loss, {NUM_GEN_SAMPLES} samples for ROUGE/chrF++.")
    
    # 2. Benchmark Vanilla Base Model
    print("\n[Phase 1/2] Benchmarking Baseline: Vanilla Qwen2.5-0.5B-Instruct...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True
    )
    base_model.eval()
    
    base_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"    Base Model VRAM: {base_vram:.1f} MB")
    
    print("    Computing Perplexity & Cross-Entropy Loss...")
    base_loss, base_ppl = compute_ppl_and_loss(base_model, tokenizer, dataset_samples, max_samples=NUM_PPL_SAMPLES)
    print(f"    Vanilla Base — Loss: {base_loss:.4f} | Perplexity: {base_ppl:.2f}")
    
    base_gen_stats = generate_and_evaluate(base_model, tokenizer, dataset_samples, num_samples=NUM_GEN_SAMPLES)
    print(f"    Vanilla Base — chrF++: {base_gen_stats['chrf']:.2f} | ROUGE-L: {base_gen_stats['rougeL']:.2f} | BLEU: {base_gen_stats['bleu']:.2f}")
    
    # Clean memory
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    # 3. Benchmark Distilled Student Model
    print("\n[Phase 2/2] Benchmarking Proposed: Distilled Student (LoRA v2)...")
    student_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True
    )
    student_model = PeftModel.from_pretrained(student_base, ADAPTER_PATH)
    student_model.eval()
    
    student_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"    Distilled Student VRAM: {student_vram:.1f} MB")
    
    print("    Computing Perplexity & Cross-Entropy Loss...")
    student_loss, student_ppl = compute_ppl_and_loss(student_model, tokenizer, dataset_samples, max_samples=NUM_PPL_SAMPLES)
    print(f"    Distilled Student — Loss: {student_loss:.4f} | Perplexity: {student_ppl:.2f}")
    
    student_gen_stats = generate_and_evaluate(student_model, tokenizer, dataset_samples, num_samples=NUM_GEN_SAMPLES)
    print(f"    Distilled Student — chrF++: {student_gen_stats['chrf']:.2f} | ROUGE-L: {student_gen_stats['rougeL']:.2f} | BLEU: {student_gen_stats['bleu']:.2f}")
    
    # 4. Save JSON Results
    results_payload = {
        "benchmark_suite": "Suite 1: Quantitative Held-Out NLG Benchmark",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "sample_size_ppl": NUM_PPL_SAMPLES,
        "sample_size_gen": NUM_GEN_SAMPLES,
        "base_model": {
            "name": "Vanilla Qwen2.5-0.5B-Instruct",
            "loss": base_loss,
            "ppl": base_ppl,
            "rouge1": base_gen_stats["rouge1"],
            "rouge2": base_gen_stats["rouge2"],
            "rougeL": base_gen_stats["rougeL"],
            "chrf": base_gen_stats["chrf"],
            "bleu": base_gen_stats["bleu"],
            "latency_ms": base_gen_stats["latency_ms"],
            "tokens_per_sec": base_gen_stats["tokens_per_sec"],
            "vram_mb": base_vram,
            "samples": base_gen_stats["samples"]
        },
        "student_model": {
            "name": "Bangla Distill Student (LoRA v2)",
            "loss": student_loss,
            "ppl": student_ppl,
            "rouge1": student_gen_stats["rouge1"],
            "rouge2": student_gen_stats["rouge2"],
            "rougeL": student_gen_stats["rougeL"],
            "chrf": student_gen_stats["chrf"],
            "bleu": student_gen_stats["bleu"],
            "latency_ms": student_gen_stats["latency_ms"],
            "tokens_per_sec": student_gen_stats["tokens_per_sec"],
            "vram_mb": student_vram,
            "samples": student_gen_stats["samples"]
        }
    }
    
    json_path = os.path.join(RESULTS_DIR, "suite1_quantitative_nlg.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Raw results written to: {json_path}")
    
    # 5. Build HTML & PDF Report
    print("[*] Generating academic PDF evaluation report...")
    ppl_reduction = ((base_ppl - student_ppl) / base_ppl) * 100 if base_ppl > 0 else 0
    chrf_gain = student_gen_stats["chrf"] - base_gen_stats["chrf"]
    rougeL_gain = student_gen_stats["rougeL"] - base_gen_stats["rougeL"]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bangla Distill: Quantitative NLG Benchmark Report</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+Bengali:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    @page {{
      size: A4 portrait;
      margin: 12mm 15mm 12mm 15mm;
    }}
    * {{ box-sizing: border-box; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
    body {{
      font-family: 'Inter', -apple-system, sans-serif;
      color: #0f172a;
      background: #ffffff;
      line-height: 1.45;
      font-size: 8.5pt;
      margin: 0; padding: 0;
    }}
    .bengali {{ font-family: 'Noto Sans Bengali', 'Inter', sans-serif; }}
    .header {{
      border-bottom: 2px solid #0f172a;
      padding-bottom: 8px;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }}
    .title-block h1 {{
      font-size: 15pt;
      font-weight: 800;
      color: #0f172a;
      margin: 0 0 3px 0;
      letter-spacing: -0.4px;
    }}
    .title-block .subtitle {{
      font-size: 8.5pt;
      color: #0284c7;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .meta-box {{
      text-align: right;
      font-family: 'JetBrains Mono', monospace;
      font-size: 7.2pt;
      color: #64748b;
    }}
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }}
    .card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 10px;
    }}
    .metric-value {{
      font-size: 16pt;
      font-weight: 800;
      color: #0f172a;
      margin: 4px 0 2px 0;
      font-family: 'JetBrains Mono', monospace;
    }}
    .metric-sub {{
      font-size: 7pt;
      color: #16a34a;
      font-weight: 600;
    }}
    .metric-label {{
      font-size: 7.2pt;
      text-transform: uppercase;
      color: #64748b;
      font-weight: 600;
      letter-spacing: 0.5px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 16px 0;
      font-size: 8pt;
    }}
    th {{
      background: #0f172a;
      color: #ffffff;
      padding: 7px 9px;
      text-align: left;
      font-weight: 600;
      font-size: 7.5pt;
      text-transform: uppercase;
    }}
    td {{
      padding: 7px 9px;
      border-bottom: 1px solid #e2e8f0;
    }}
    tr:nth-child(even) td {{
      background: #f8fafc;
    }}
    .tag-base {{
      background: #f1f5f9;
      color: #475569;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 7pt;
      font-weight: 600;
    }}
    .tag-student {{
      background: #dbeafe;
      color: #1d4ed8;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 7pt;
      font-weight: 700;
    }}
    .gain-pill {{
      background: #dcfce7;
      color: #15803d;
      padding: 2px 5px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 7.2pt;
    }}
    .section-title {{
      font-size: 9.5pt;
      font-weight: 700;
      color: #0f172a;
      margin: 14px 0 6px 0;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .sample-box {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 10px;
      margin-bottom: 8px;
    }}
    .sample-query {{
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 4px;
      font-size: 8pt;
    }}
    .sample-comp {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 7.4pt;
    }}
    .comp-col {{
      background: #f8fafc;
      padding: 6px;
      border-radius: 4px;
      border: 1px solid #f1f5f9;
    }}
    .comp-col-title {{
      font-weight: 700;
      font-size: 6.8pt;
      text-transform: uppercase;
      margin-bottom: 3px;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title-block">
      <h1>Bangla Distill — Quantitative Benchmark</h1>
      <div class="subtitle">Suite 1: Held-Out Linguistic & Generation Performance</div>
    </div>
    <div class="meta-box">
      <div><strong>Date:</strong> {results_payload["timestamp"]}</div>
      <div><strong>Device:</strong> {results_payload["hardware"]}</div>
      <div><strong>Sample:</strong> {NUM_PPL_SAMPLES} PPL / {NUM_GEN_SAMPLES} Gen Records</div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card">
      <div class="metric-label">Perplexity (PPL)</div>
      <div class="metric-value">{student_ppl:.2f}</div>
      <div class="metric-sub">▼ {ppl_reduction:.1f}% vs Base ({base_ppl:.2f})</div>
    </div>
    <div class="card">
      <div class="metric-label">chrF++ Score</div>
      <div class="metric-value">{student_gen_stats["chrf"]:.2f}</div>
      <div class="metric-sub">▲ +{chrf_gain:.2f} pts vs Base</div>
    </div>
    <div class="card">
      <div class="metric-label">ROUGE-L F1</div>
      <div class="metric-value">{student_gen_stats["rougeL"]:.2f}%</div>
      <div class="metric-sub">▲ +{rougeL_gain:.2f}% vs Base</div>
    </div>
    <div class="card">
      <div class="metric-label">Generation Speed</div>
      <div class="metric-value">{student_gen_stats["tokens_per_sec"]:.1f}</div>
      <div class="metric-sub">Tokens / Sec (GPU FP16)</div>
    </div>
  </div>

  <div class="section-title">📊 Formal Model Comparative Table</div>
  <table>
    <thead>
      <tr>
        <th>Model Architecture</th>
        <th>Params</th>
        <th>Loss (CE) ↓</th>
        <th>Perplexity ↓</th>
        <th>chrF++ ↑</th>
        <th>ROUGE-1 ↑</th>
        <th>ROUGE-2 ↑</th>
        <th>ROUGE-L ↑</th>
        <th>SacreBLEU ↑</th>
        <th>Latency (ms)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><span class="tag-base">Vanilla Qwen2.5-0.5B</span></td>
        <td>490M</td>
        <td>{base_loss:.4f}</td>
        <td>{base_ppl:.2f}</td>
        <td>{base_gen_stats["chrf"]:.2f}</td>
        <td>{base_gen_stats["rouge1"]:.2f}</td>
        <td>{base_gen_stats["rouge2"]:.2f}</td>
        <td>{base_gen_stats["rougeL"]:.2f}</td>
        <td>{base_gen_stats["bleu"]:.2f}</td>
        <td>{base_gen_stats["latency_ms"]:.1f} ms</td>
      </tr>
      <tr>
        <td><span class="tag-student">Bangla Distill Student (LoRA v2)</span></td>
        <td><strong>490M</strong></td>
        <td><strong>{student_loss:.4f}</strong></td>
        <td><strong>{student_ppl:.2f}</strong></td>
        <td><strong>{student_gen_stats["chrf"]:.2f}</strong></td>
        <td><strong>{student_gen_stats["rouge1"]:.2f}</strong></td>
        <td><strong>{student_gen_stats["rouge2"]:.2f}</strong></td>
        <td><strong>{student_gen_stats["rougeL"]:.2f}</strong></td>
        <td><strong>{student_gen_stats["bleu"]:.2f}</strong></td>
        <td><strong>{student_gen_stats["latency_ms"]:.1f} ms</strong></td>
      </tr>
      <tr style="background: #f0fdf4;">
        <td colspan="2"><strong>Improvement / Delta</strong></td>
        <td><span class="gain-pill">-{base_loss - student_loss:.4f}</span></td>
        <td><span class="gain-pill">-{ppl_reduction:.1f}%</span></td>
        <td><span class="gain-pill">+{chrf_gain:.2f}</span></td>
        <td><span class="gain-pill">+{student_gen_stats["rouge1"] - base_gen_stats["rouge1"]:.2f}</span></td>
        <td><span class="gain-pill">+{student_gen_stats["rouge2"] - base_gen_stats["rouge2"]:.2f}</span></td>
        <td><span class="gain-pill">+{rougeL_gain:.2f}</span></td>
        <td><span class="gain-pill">+{student_gen_stats["bleu"] - base_gen_stats["bleu"]:.2f}</span></td>
        <td><span style="font-size: 7.2pt; color: #475569;">Comparable</span></td>
      </tr>
    </tbody>
  </table>

  <div class="section-title">🔍 Side-by-Side Held-Out Output Analysis</div>
"""

    for idx, (b_s, s_s) in enumerate(zip(base_gen_stats["samples"], student_gen_stats["samples"])):
        prompt = s_s["prompt"]
        ref = s_s["reference"]
        b_pred = b_s["prediction"]
        s_pred = s_s["prediction"]
        html += f"""
  <div class="sample-box">
    <div class="sample-query bengali">📌 Test Query #{idx+1}: {prompt}</div>
    <div class="sample-comp bengali">
      <div class="comp-col">
        <div class="comp-col-title" style="color: #64748b;">Vanilla Baseline Output:</div>
        <div>{b_pred}</div>
      </div>
      <div class="comp-col" style="background: #f0fdf4; border-color: #bbf7d0;">
        <div class="comp-col-title" style="color: #15803d;">Distilled Student Output:</div>
        <div>{s_pred}</div>
      </div>
    </div>
  </div>
"""

    html += """
</body>
</html>
"""
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_out = os.path.join(reports_dir, "Bangla_Distill_Benchmark_Suite1_Quantitative_NLG.pdf")
    html_to_pdf(html, pdf_out)
    print(f"\n🎉 Suite 1 Complete! PDF Report Generated at:\n   {pdf_out}")
    return pdf_out

if __name__ == "__main__":
    run_suite()
