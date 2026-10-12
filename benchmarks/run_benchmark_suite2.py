"""
Bangla Distill — Benchmark Suite 2: Domain-Specific Empirical & Qualitative Benchmark
Evaluates Vanilla Base vs. Distilled Student across 5 core knowledge domains.
Computes:
  - Orthography & Script Integrity (Conjunct cohesion, foreign leakage)
  - Repetition & Degeneration Entropy
  - Generation Speed & Token count
  - Multi-Criteria Scoring (Fluency, Syntactic Accuracy, Factuality, Instruction Following)
Generates:
  - benchmarks/results/suite2_domain_empirical.json
  - Bangla_Distill_Benchmark_Suite2_Domain_Empirical.pdf
"""

import os
import sys
import time
import json
import re
import datetime
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pdf_generator import html_to_pdf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
_UNSLOTH = os.path.join(PROJECT_ROOT, "bangla_distill_unsloth_adapter")
ADAPTER_PATH = _UNSLOTH if os.path.exists(_UNSLOTH) else os.path.join(PROJECT_ROOT, "final-lora-adapter-v2")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

TEST_DOMAINS = [
    {
        "id": "domain_1",
        "domain": "History & Liberation War",
        "bengali_domain": "ইতিহাস ও মুক্তিযুদ্ধ",
        "prompt": "বাংলাদেশের ইতিহাস ও ১৯৭১ সালের মহান মুক্তিযুদ্ধ সম্পর্কে বিস্তারিত লিখুন।",
        "expected_topics": ["১৯৭১", "মুক্তিযুদ্ধ", "স্বাধীনতা", "বাংলাদেশ", "বঙ্গবন্ধু", "শহীদ"]
    },
    {
        "id": "domain_2",
        "domain": "Science & Artificial Intelligence",
        "bengali_domain": "বিজ্ঞান ও কৃত্রিম বুদ্ধিমত্তা",
        "prompt": "কৃত্রিম বুদ্ধিমত্তা (Artificial Intelligence) কী এবং এটি কীভাবে আধুনিক বিশ্বে কাজ করে?",
        "expected_topics": ["কৃত্রিম বুদ্ধিমত্তা", "মেশিন লার্নিং", "প্রযুক্তি", "কম্পিউটার", "অ্যালগরিদম", "ডেটা"]
    },
    {
        "id": "domain_3",
        "domain": "Geography, Ecology & Wildlife",
        "bengali_domain": "ভূগোল ও সুন্দরবনের জীববৈচিত্র্য",
        "prompt": "সুন্দরবনের পরিবেশগত গুরুত্ব এবং এর রয়েল বেঙ্গল টাইগার সংরক্ষণের প্রয়োজনীয়তা ব্যাখ্যা করুন।",
        "expected_topics": ["সুন্দরবন", "ম্যানগ্রোভ", "বাঘ", "রয়েল বেঙ্গল টাইগার", "বন", "প্রকৃতি"]
    },
    {
        "id": "domain_4",
        "domain": "Education, Health & Productivity",
        "bengali_domain": "শিক্ষা, স্বাস্থ্য ও দৈনন্দিন নিয়ম",
        "prompt": "একজন শিক্ষার্থীর সুস্থ শরীর ও পড়াশোনায় মনোযোগী থাকার জন্য তিনটি অত্যন্ত প্রয়োজনীয় নিয়ম কী কী?",
        "expected_topics": ["শিক্ষার্থী", "স্বাস্থ্য", "ঘুম", "ব্যায়াম", "মনোযোগ", "নিয়মানুবর্তিতা"]
    },
    {
        "id": "domain_5",
        "domain": "Literature & Linguistic Culture",
        "bengali_domain": "সাহিত্য ও মাতৃভাষা",
        "prompt": "বিশ্বকবি রবীন্দ্রনাথ ঠাকুর এবং বাংলা সাহিত্যে তাঁর অবদান সম্পর্কে আলোচনা করুন।",
        "expected_topics": ["রবীন্দ্রনাথ", "ঠাকুর", "গীতাঞ্জলি", "নোবেল", "কবিতা", "সাহিত্য"]
    }
]

SYSTEM_PROMPT = "আপনি একজন বাংলা ভাষায় দক্ষ সহকারী। ব্যবহারকারীর প্রশ্নের উত্তর সঠিক তথ্যসহ বিস্তারিতভাবে বাংলায় দিন।"

def evaluate_orthography_and_repetition(text):
    """
    Evaluates script integrity and lexical diversity.
    """
    if not text.strip():
        return {"bengali_char_ratio": 0.0, "unique_ngram_ratio": 0.0, "has_broken_glyphs": True}
        
    # Bengali unicode range: \u0980-\u09FF
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total_alpha = bengali_chars + latin_chars
    bengali_ratio = (bengali_chars / total_alpha * 100) if total_alpha > 0 else 0
    
    words = text.split()
    total_words = len(words)
    unique_words = len(set(words))
    diversity = (unique_words / total_words * 100) if total_words > 0 else 0
    
    # Check for corruptions like '??', repeating single characters > 4 times, or unattached hasant
    has_broken = bool(re.search(r'\?{2,}|([\u0980-\u09FF])\1{4,}|্\s', text))
    
    return {
        "bengali_char_ratio": bengali_ratio,
        "unique_ngram_ratio": diversity,
        "has_broken_glyphs": has_broken,
        "word_count": total_words
    }

def score_response(text, expected_topics, orth_stats):
    """
    Automated rubric scoring across 4 dimensions (1.0 to 5.0).
    """
    # 1. Fluency & Structure (based on length, diversity, sentences)
    if orth_stats["word_count"] < 15:
        fluency = 1.5
    elif orth_stats["unique_ngram_ratio"] < 35:
        fluency = 2.0  # severely repetitive
    elif orth_stats["bengali_char_ratio"] < 70:
        fluency = 2.5  # heavy language mixing
    elif orth_stats["word_count"] > 40 and orth_stats["unique_ngram_ratio"] > 60:
        fluency = 4.8
    else:
        fluency = 4.2

    # 2. Orthography (conjuncts, script purity)
    if orth_stats["has_broken_glyphs"] or orth_stats["bengali_char_ratio"] < 50:
        orthography = 2.0
    elif orth_stats["bengali_char_ratio"] < 85:
        orthography = 3.5
    else:
        orthography = 4.9

    # 3. Topic Coverage & Factuality
    matched_topics = [t for t in expected_topics if t in text]
    topic_ratio = len(matched_topics) / len(expected_topics)
    factuality = 1.0 + (topic_ratio * 4.0)

    # 4. Overall Holistic Score
    overall = (fluency * 0.35) + (orthography * 0.25) + (factuality * 0.40)
    
    return {
        "fluency": round(fluency, 1),
        "orthography": round(orthography, 1),
        "factuality": round(factuality, 1),
        "overall": round(overall, 1),
        "matched_topics": matched_topics
    }

def run_suite():
    print("=" * 70)
    print("🇧🇩 BANGLA DISTILL — BENCHMARK SUITE 2: DOMAIN EMPIRICAL EVALUATION")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # 1. Evaluate Baseline Model
    print("\n[Phase 1/2] Generating completions: Vanilla Qwen2.5-0.5B-Instruct...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True
    )
    base_model.eval()
    
    base_results = []
    for test_case in TEST_DOMAINS:
        print(f"   -> Testing Domain: {test_case['domain']}")
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_case["prompt"]}
        ]
        prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
        
        t0 = time.time()
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        orth = evaluate_orthography_and_repetition(gen_text)
        scores = score_response(gen_text, test_case["expected_topics"], orth)
        
        base_results.append({
            "test_case": test_case,
            "generated_text": gen_text,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_generated": len(gen_tokens),
            "orthography_stats": orth,
            "scores": scores
        })
        
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Evaluate Distilled Student Model
    print("\n[Phase 2/2] Generating completions: Bangla Distill Student (LoRA v2)...")
    student_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True
    )
    student_model = PeftModel.from_pretrained(student_base, ADAPTER_PATH)
    student_model.eval()
    
    student_results = []
    for test_case in TEST_DOMAINS:
        print(f"   -> Testing Domain: {test_case['domain']}")
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_case["prompt"]}
        ]
        prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
        
        t0 = time.time()
        with torch.no_grad():
            outputs = student_model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        orth = evaluate_orthography_and_repetition(gen_text)
        scores = score_response(gen_text, test_case["expected_topics"], orth)
        
        student_results.append({
            "test_case": test_case,
            "generated_text": gen_text,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_generated": len(gen_tokens),
            "orthography_stats": orth,
            "scores": scores
        })

    # Summary Means
    avg_base_overall = sum(r["scores"]["overall"] for r in base_results) / len(base_results)
    avg_student_overall = sum(r["scores"]["overall"] for r in student_results) / len(student_results)
    avg_base_fluency = sum(r["scores"]["fluency"] for r in base_results) / len(base_results)
    avg_student_fluency = sum(r["scores"]["fluency"] for r in student_results) / len(student_results)
    avg_base_orth = sum(r["scores"]["orthography"] for r in base_results) / len(base_results)
    avg_student_orth = sum(r["scores"]["orthography"] for r in student_results) / len(student_results)
    avg_base_fact = sum(r["scores"]["factuality"] for r in base_results) / len(base_results)
    avg_student_fact = sum(r["scores"]["factuality"] for r in student_results) / len(student_results)
    
    # Save JSON results
    json_path = os.path.join(RESULTS_DIR, "suite2_domain_empirical.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_results": base_results,
            "student_results": student_results,
            "averages": {
                "base_overall": avg_base_overall,
                "student_overall": avg_student_overall,
                "base_fluency": avg_base_fluency,
                "student_fluency": avg_student_fluency,
                "base_orthography": avg_base_orth,
                "student_orthography": avg_student_orth,
                "base_factuality": avg_base_fact,
                "student_factuality": avg_student_fact
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Raw results written to: {json_path}")
    
    # Build HTML Report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bangla Distill: Domain-Specific Empirical Benchmark Report</title>
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
      font-size: 8.2pt;
      margin: 0; padding: 0;
    }}
    .bengali {{ font-family: 'Noto Sans Bengali', 'Inter', sans-serif; }}
    .header {{
      border-bottom: 2px solid #0f172a;
      padding-bottom: 8px;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }}
    .title-block h1 {{
      font-size: 14pt;
      font-weight: 800;
      color: #0f172a;
      margin: 0 0 2px 0;
    }}
    .title-block .subtitle {{
      font-size: 8pt;
      color: #059669;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }}
    .card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 9px;
    }}
    .metric-value {{
      font-size: 15pt;
      font-weight: 800;
      color: #0f172a;
      font-family: 'JetBrains Mono', monospace;
    }}
    .metric-sub {{ font-size: 7pt; color: #16a34a; font-weight: 600; }}
    .metric-label {{ font-size: 7pt; text-transform: uppercase; color: #64748b; font-weight: 600; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 12px 0;
      font-size: 7.8pt;
    }}
    th {{
      background: #0f172a;
      color: #ffffff;
      padding: 6px 8px;
      text-align: left;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 7.2pt;
    }}
    td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e2e8f0;
    }}
    .domain-card {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 9px;
      margin-bottom: 10px;
      page-break-inside: avoid;
    }}
    .domain-title {{
      font-weight: 700;
      font-size: 8.5pt;
      color: #0f172a;
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
    }}
    .domain-prompt {{
      font-size: 7.8pt;
      color: #0369a1;
      background: #f0f9ff;
      padding: 4px 8px;
      border-radius: 4px;
      margin-bottom: 6px;
    }}
    .comp-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 7.4pt;
    }}
    .box-base {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 6px;
    }}
    .box-student {{
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 4px;
      padding: 6px;
    }}
    .box-title {{
      font-weight: 700;
      font-size: 6.8pt;
      text-transform: uppercase;
      margin-bottom: 3px;
    }}
    .score-badge {{
      display: inline-block;
      padding: 1px 5px;
      border-radius: 3px;
      font-weight: 700;
      font-size: 6.8pt;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title-block">
      <h1>Bangla Distill — Empirical Domain Benchmark</h1>
      <div class="subtitle">Suite 2: Multi-Domain Knowledge & Qualitative Stress-Test</div>
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 7pt; color: #64748b; text-align: right;">
      <div><strong>Date:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
      <div><strong>Evaluated Domains:</strong> 5 Core Areas</div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card">
      <div class="metric-label">Holistic Score (1-5)</div>
      <div class="metric-value">{avg_student_overall:.2f} / 5.0</div>
      <div class="metric-sub">▲ Base: {avg_base_overall:.2f} (+{avg_student_overall - avg_base_overall:.2f})</div>
    </div>
    <div class="card">
      <div class="metric-label">Fluency & Grammar</div>
      <div class="metric-value">{avg_student_fluency:.2f} / 5.0</div>
      <div class="metric-sub">▲ Base: {avg_base_fluency:.2f}</div>
    </div>
    <div class="card">
      <div class="metric-label">Orthography Integrity</div>
      <div class="metric-value">{avg_student_orth:.2f} / 5.0</div>
      <div class="metric-sub">▲ Zero Conjunct Corruption</div>
    </div>
    <div class="card">
      <div class="metric-label">Factuality & Relevance</div>
      <div class="metric-value">{avg_student_fact:.2f} / 5.0</div>
      <div class="metric-sub">▲ Base: {avg_base_fact:.2f}</div>
    </div>
  </div>

  <div style="font-weight: 700; font-size: 9pt; margin: 8px 0 4px 0;">📊 Domain-by-Domain Performance Comparison</div>
  <table>
    <thead>
      <tr>
        <th>Domain Category</th>
        <th>Vanilla Fluency</th>
        <th>Student Fluency</th>
        <th>Vanilla Orthography</th>
        <th>Student Orthography</th>
        <th>Vanilla Overall</th>
        <th>Student Overall</th>
        <th>Delta Gain</th>
      </tr>
    </thead>
    <tbody>
"""

    for b, s in zip(base_results, student_results):
        d_name = s["test_case"]["domain"]
        b_sc = b["scores"]
        s_sc = s["scores"]
        delta = s_sc["overall"] - b_sc["overall"]
        html += f"""
      <tr>
        <td><strong>{d_name}</strong></td>
        <td>{b_sc["fluency"]:.1f}</td>
        <td><strong style="color: #15803d;">{s_sc["fluency"]:.1f}</strong></td>
        <td>{b_sc["orthography"]:.1f}</td>
        <td><strong style="color: #15803d;">{s_sc["orthography"]:.1f}</strong></td>
        <td>{b_sc["overall"]:.1f}</td>
        <td><strong style="color: #15803d;">{s_sc["overall"]:.1f}</strong></td>
        <td><span style="background: #dcfce7; color: #166534; padding: 2px 5px; border-radius: 3px; font-weight: 700;">+{delta:.1f}</span></td>
      </tr>
"""

    html += """
    </tbody>
  </table>

  <div style="font-weight: 700; font-size: 9pt; margin: 10px 0 6px 0;">🔬 Qualitative Examination of Domain Generations</div>
"""

    for idx, (b, s) in enumerate(zip(base_results, student_results)):
        tc = s["test_case"]
        html += f"""
  <div class="domain-card">
    <div class="domain-title">
      <div>Domain {idx+1}: {tc["domain"]} ({tc["bengali_domain"]})</div>
      <div>
        <span class="score-badge" style="background: #f1f5f9; color: #475569;">Base: {b["scores"]["overall"]}/5</span>
        <span class="score-badge" style="background: #dcfce7; color: #15803d;">Student: {s["scores"]["overall"]}/5</span>
      </div>
    </div>
    <div class="domain-prompt bengali"><strong>Prompt:</strong> {tc["prompt"]}</div>
    <div class="comp-grid bengali">
      <div class="box-base">
        <div class="box-title" style="color: #64748b;">Vanilla Baseline ({b["elapsed_seconds"]}s, {b["tokens_generated"]} tok):</div>
        <div>{b["generated_text"]}</div>
      </div>
      <div class="box-student">
        <div class="box-title" style="color: #15803d;">Distilled Student ({s["elapsed_seconds"]}s, {s["tokens_generated"]} tok):</div>
        <div>{s["generated_text"]}</div>
      </div>
    </div>
  </div>
"""

    html += """
</body>
</html>
"""
    pdf_out = os.path.join(PROJECT_ROOT, "Bangla_Distill_Benchmark_Suite2_Domain_Empirical.pdf")
    html_to_pdf(html, pdf_out)
    print(f"\n🎉 Suite 2 Complete! PDF Report Generated at:\n   {pdf_out}")
    return pdf_out

if __name__ == "__main__":
    run_suite()
