# 🇧🇩 Bangla Distill

> **Transferring High-Fidelity Bengali Natural Language Generation from Frontier Teachers into an Ultra-Compact 500M Student Architecture**

[![Base Model](https://img.shields.io/badge/Base_Model-Qwen2.5--0.5B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![Teacher Models](https://img.shields.io/badge/Teachers-Gemma--2--9B--IT_%2B_Qwen--3.5--9B-purple)](#teacher-models)
[![Quantization](https://img.shields.io/badge/Method-QLoRA_(4--bit_NF4)-green)](#methodology)
[![Adapter Size](https://img.shields.io/badge/LoRA_Adapter-~35_MB-orange)](#model-deliverables)

---

## 📌 Overview

Bengali has over 230 million native speakers, yet open-source language models under 1-billion parameters frequently suffer from severe script fragmentation, unnatural grammar, and high perplexity. Deploying 7B+ models is impractical for resource-constrained or edge environments.

**Bangla Distill** addresses this challenge via an **offline dual-teacher knowledge distillation** framework combined with **QLoRA (Parameter-Efficient Fine-Tuning)** to supervise an ultra-compact student model: **Qwen2.5-0.5B-Instruct**. The resulting fine-tuned model delivers fluent Bengali text generation encapsulated in an ultra-lightweight LoRA adapter runnable on consumer GPUs (<6 GB VRAM) and free-tier Google Colab/Kaggle T4 GPUs.

---

## 🏗️ Architecture & Methodology

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontier Teacher Ensembles                  │
│   Gemma-2-9B-IT (Reasoning & Exposition)  +  Qwen-3.5-9B (Morphology)  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼ Curated Teacher Supervision (80,000+ pairs)
┌─────────────────────────────────────────────────────────────────┐
│                    Offline Bengali Corpus                       │
│        teacher_corpus_gemma.jsonl  +  teacher_corpus_qwen.jsonl │
│                     train_chat.jsonl (5,387)                    │
│                      val_chat.jsonl (598)                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼ QLoRA SFT (ChatML Completion-Only Loss)
┌─────────────────────────────────────────────────────────────────┐
│                   Compact Student Model (0.5B)                  │
│                      Qwen2.5-0.5B-Instruct                      │
│                    + final-lora-adapter-v2                      │
└─────────────────────────────────────────────────────────────────┘
```

1. **Dual-Teacher Supervision**: Combines Gemma-2-9B-IT (structured reasoning, domain-specific exposition) with Qwen-3.5-9B (morphological nuance and script cohesion) to eliminate single-teacher stylistic quirks and hallucinations.
2. **Student Backbone**: `Qwen/Qwen2.5-0.5B-Instruct` (~500M parameters) with a 151k+ vocabulary tokenizer to avoid out-of-vocabulary subword fragmentation in Bengali script.
3. **PEFT with QLoRA (v2)**:
   - Base model quantized in 4-bit NormalFloat (NF4).
   - LoRA targeting all 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Hyperparameters: Rank $r=16$, Alpha $\alpha=32$, Dropout $0.05$.
   - Loss calculation: ChatML completion-only masking (assistant tokens only).

---

## 📁 Repository Structure

```
Bangla Distill/
├── final-lora-adapter-v2/              # Production LoRA adapter weights (v2)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── chat_template.jinja
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── TRAINING_INFO.txt
├── train_bangla_unsloth_kaggle.ipynb   # ⚡ Fast Unsloth SFT + DPO training on Kaggle GPU
├── run_bangla_student_unsloth.ipynb    # ⚡ Unsloth accelerated inference runner on Kaggle
├── train_unsloth_kaggle.py             # Standalone Unsloth training script
├── ui/                                 # FastAPI + HTML5 Interactive Chat Console
│   ├── app.py                          # Backend API & inference engine
│   ├── launch_ui.bat / launch_ui.ps1
│   └── static/                         # Frontend UI (HTML, CSS, JS)
├── run_local_inference.py              # CLI GPU inference script
├── prepare_chat_dataset.py             # Data preparation & format pipeline
├── run_bangla_student.ipynb            # Colab/Kaggle inference notebook
├── train_bangla_v2.ipynb               # Colab/Kaggle training notebook (v2)
├── teacher_corpus_gemma.jsonl          # Distilled corpus from Gemma-2-9B (40k samples)
├── teacher_corpus_qwen.jsonl           # Distilled corpus from Qwen-3.5-9B (40k samples)
├── train_chat.jsonl                    # Train split (ChatML format)
├── val_chat.jsonl                      # Validation split (ChatML format)
├── Bangla_Distill_Faculty_Overview.pdf # Faculty briefing report
├── Bangla_Distill_Project_Report.pdf   # Complete project documentation
├── local_test_results.md               # Empirical evaluation logs
└── START_UI.bat                        # One-click UI startup script
```

---

## 🚀 Quickstart & Inference

### 1. Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (or CPU fallback)
- PyTorch, Transformers, PEFT, BitsAndBytes, FastAPI, Uvicorn

```bash
pip install torch transformers peft accelerate bitsandbytes fastapi uvicorn python-dotenv
```

### 2. Local CLI Inference

Run interactive chat or evaluate a single prompt:

```bash
# Interactive Chat Mode
python run_local_inference.py --interactive

# Single Prompt Evaluation
python run_local_inference.py --prompt "বাংলাদেশের স্বাধীনতা যুদ্ধের তাৎপর্য ব্যাখ্যা করুন।"
```

### 3. Interactive Web Chat Interface

Launch the dedicated web application:

```bash
# Windows Quick Launch
START_UI.bat

# Or manual launch
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8000
```
Open your browser at `http://127.0.0.1:8000`.

---

## 📊 Empirical Evaluation

| Test Prompt | Vanilla Base (Qwen2.5-0.5B) | Distilled Student (LoRA v2) |
| :--- | :--- | :--- |
| **বাংলাদেশের ইতিহাস ও মুক্তিযুদ্ধ** | Mixed language tokens, grammatical fragmentation | Coherent Bengali text, correct historical terminology |
| **কৃত্রিম বুদ্ধিমত্তা কী ও এর কাজ** | Incoherent rambling, high perplexity | Clear definition, functional explanation in formal Bengali |

---

## ⚡ Training on Kaggle with Unsloth (Recommended)

To retrain the student model and run the DPO reinforcement stage for free using Kaggle's GPU:

1. **Go to Kaggle Notebooks:** Create a new notebook on [kaggle.com/code](https://www.kaggle.com/code).
2. **Import Notebook:** Click `File` ➔ `Import Notebook` ➔ Upload [`train_bangla_unsloth_kaggle.ipynb`](file:///d:/Bangla%20Distill/train_bangla_unsloth_kaggle.ipynb).
3. **Enable GPU & Internet:**
   - On the right sidebar, open **Settings (⚙️)**.
   - Set **Accelerator** to **GPU P100** or **GPU T4 x2**.
   - Set **Internet** to **On**.
4. **Run All Cells:** The notebook automatically fetches `train_chat.jsonl` and `val_chat.jsonl`, performs 4-bit Unsloth QLoRA SFT training, runs DPO preference alignment, and packages your adapter into a downloadable zip file in `/kaggle/working/`.

---

## 📄 License & Attribution

- **Base Model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Tongyi Qianwen / Alibaba Cloud)
- **Adapter Weights & Training Pipeline:** Bangla Distill Project
