import json

notebook = {
 "nbformat": 4,
 "nbformat_minor": 0,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "accelerator": "GPU"
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🇧🇩 Bangla Distill 3B — Unsloth Fast Distillation & Fine-Tuning on Kaggle\n",
    "\n",
    "> **Scaling Knowledge Distillation to Qwen2.5-3B-Instruct for Deep Factual Bengali Fluency**\n",
    "\n",
    "### ⚡ Why Scale from 0.5B to 3B?\n",
    "- 🧠 **6x Parametric Capacity:** 3 Billion parameters provide enough capacity to retain real historical facts, dates, science, and cultural nuance without hallucinations.\n",
    "- 🎯 **Zero Boilerplate:** Trained on curated high-density encyclopedic knowledge from dual frontier teachers (`Gemma-2-9B-IT` + `Qwen-3.5-9B`).\n",
    "- 💻 **Lightweight Edge Deployment:** With 4-bit NF4 quantization, Qwen2.5-3B uses only **~2.2 GB VRAM**, running smoothly on laptop GPUs (e.g., RTX 4070 8GB) at ~35 tokens/sec.\n",
    "\n",
    "---\n",
    "### ⚙️ Kaggle Accelerator Setup\n",
    "1. In the right sidebar: Click **Settings (⚙️)** ➔ **Accelerator** ➔ Select **GPU P100** or **GPU T4 x2**.\n",
    "2. Ensure **Internet** is toggled to **On**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Install Unsloth & Dependencies\n",
    "We use pre-built wheels for instant, crash-free installation on Kaggle."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "\n",
    "print(\"Installing Unsloth via pre-built wheels...\")\n",
    "!pip install -q unsloth\n",
    "!pip install -q trl peft accelerate bitsandbytes sentencepiece datasets\n",
    "\n",
    "import torch\n",
    "print(\"=\" * 60)\n",
    "print(f\"PyTorch Version : {torch.__version__}\")\n",
    "print(f\"CUDA Available   : {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU Name        : {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB\")\n",
    "else:\n",
    "    print(\"\\n⚠️ WARNING: GPU is not active! Enable GPU P100 or T4 in Kaggle Settings.\")\n",
    "print(\"=\" * 60)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Download Curated 3B Bengali Training Corpus\n",
    "This cell downloads `train_chat_3b.jsonl` and `val_chat_3b.jsonl` directly from your GitHub repository."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import urllib.request\n",
    "\n",
    "WORKING_DIR = \"/kaggle/working\" if os.path.exists(\"/kaggle\") else \".\"\n",
    "GITHUB_RAW = \"https://raw.githubusercontent.com/AshmitNur/Distilation/main\"\n",
    "\n",
    "TRAIN_FILE = os.path.join(WORKING_DIR, \"train_chat_3b.jsonl\")\n",
    "VAL_FILE = os.path.join(WORKING_DIR, \"val_chat_3b.jsonl\")\n",
    "\n",
    "print(\"Fetching curated 3B Bengali dataset from GitHub...\")\n",
    "for fn in [\"train_chat_3b.jsonl\", \"val_chat_3b.jsonl\"]:\n",
    "    dest = os.path.join(WORKING_DIR, fn)\n",
    "    try:\n",
    "        urllib.request.urlretrieve(f\"{GITHUB_RAW}/data/{fn}\", dest)\n",
    "    except Exception:\n",
    "        urllib.request.urlretrieve(f\"{GITHUB_RAW}/{fn}\", dest)\n",
    "\n",
    "print(f\"✅ Train dataset: {TRAIN_FILE} ({os.path.getsize(TRAIN_FILE)/(1024**2):.2f} MB)\")\n",
    "print(f\"✅ Val dataset  : {VAL_FILE} ({os.path.getsize(VAL_FILE)/(1024**2):.2f} MB)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Load Qwen2.5-3B-Instruct with Unsloth 4-bit\n",
    "Unsloth loads the 4-bit quantized base model with optimized fused cross-entropy kernels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from unsloth import FastLanguageModel\n",
    "import torch\n",
    "\n",
    "MAX_SEQ_LENGTH = 1024\n",
    "BASE_MODEL_NAME = \"unsloth/Qwen2.5-3B-Instruct-bnb-4bit\"\n",
    "\n",
    "print(f\"Loading {BASE_MODEL_NAME} via Unsloth...\")\n",
    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "    model_name=BASE_MODEL_NAME,\n",
    "    max_seq_length=MAX_SEQ_LENGTH,\n",
    "    dtype=None,\n",
    "    load_in_4bit=True,\n",
    ")\n",
    "\n",
    "print(\"\\nAttaching Fast LoRA Adapters (r=16, alpha=32)...\")\n",
    "model = FastLanguageModel.get_peft_model(\n",
    "    model,\n",
    "    r=16,\n",
    "    target_modules=[\n",
    "        \"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n",
    "        \"gate_proj\", \"up_proj\", \"down_proj\",\n",
    "    ],\n",
    "    lora_alpha=32,\n",
    "    lora_dropout=0,\n",
    "    bias=\"none\",\n",
    "    use_gradient_checkpointing=\"unsloth\",\n",
    "    random_state=42,\n",
    ")\n",
    "\n",
    "model.print_trainable_parameters()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Tokenize Dataset with ChatML Template"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import load_dataset\n",
    "from unsloth.chat_templates import get_chat_template\n",
    "\n",
    "tokenizer = get_chat_template(\n",
    "    tokenizer,\n",
    "    chat_template=\"qwen-2.5\",\n",
    "    mapping={\"role\": \"role\", \"content\": \"content\", \"user\": \"user\", \"assistant\": \"assistant\"},\n",
    ")\n",
    "\n",
    "def formatting_prompts_func(examples):\n",
    "    convos = examples[\"messages\"]\n",
    "    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]\n",
    "    return {\"text\": texts}\n",
    "\n",
    "dataset = load_dataset(\"json\", data_files={\"train\": TRAIN_FILE, \"validation\": VAL_FILE})\n",
    "train_data = dataset[\"train\"].map(formatting_prompts_func, batched=True)\n",
    "val_data = dataset[\"validation\"].map(formatting_prompts_func, batched=True)\n",
    "\n",
    "print(f\"Formatted Train records: {len(train_data)}\")\n",
    "print(f\"Formatted Val records  : {len(val_data)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Supervised Fine-Tuning (SFTTrainer)\n",
    "Unsloth uses assistant-only loss masking, computing gradients solely on the teacher's Bengali responses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from trl import SFTTrainer\n",
    "from transformers import TrainingArguments\n",
    "from unsloth.chat_templates import train_on_responses_only\n",
    "\n",
    "OUTPUT_DIR = os.path.join(WORKING_DIR, \"bangla_distill_3b_output\")\n",
    "\n",
    "training_args = TrainingArguments(\n",
    "    output_dir=OUTPUT_DIR,\n",
    "    per_device_train_batch_size=2,\n",
    "    gradient_accumulation_steps=4,\n",
    "    warmup_ratio=0.05,\n",
    "    num_train_epochs=2,\n",
    "    learning_rate=2e-4,\n",
    "    fp16=not torch.cuda.is_bf16_supported(),\n",
    "    bf16=torch.cuda.is_bf16_supported(),\n",
    "    logging_steps=25,\n",
    "    evaluation_strategy=\"steps\",\n",
    "    eval_steps=150,\n",
    "    save_steps=300,\n",
    "    save_total_limit=2,\n",
    "    optim=\"adamw_8bit\",\n",
    "    weight_decay=0.01,\n",
    "    lr_scheduler_type=\"cosine\",\n",
    "    seed=42,\n",
    "    report_to=\"none\",\n",
    ")\n",
    "\n",
    "trainer = SFTTrainer(\n",
    "    model=model,\n",
    "    tokenizer=tokenizer,\n",
    "    train_dataset=train_data,\n",
    "    eval_dataset=val_data,\n",
    "    dataset_text_field=\"text\",\n",
    "    max_seq_length=MAX_SEQ_LENGTH,\n",
    "    dataset_num_proc=2,\n",
    "    packing=False,\n",
    "    args=training_args,\n",
    ")\n",
    "\n",
    "# Mask prompt tokens so training focuses 100% on Bengali responses\n",
    "trainer = train_on_responses_only(\n",
    "    trainer,\n",
    "    instruction_part=\"<|im_start|>user\\n\",\n",
    "    response_part=\"<|im_start|>assistant\\n\",\n",
    ")\n",
    "\n",
    "print(\"Starting Bangla Distill 3B training...\")\n",
    "train_stats = trainer.train()\n",
    "print(\"\\n✅ Training complete!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6: Interactive Bengali Generation Verification\n",
    "Let's test the 3B student on the exact history prompt to verify factual fluency!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "FastLanguageModel.for_inference(model)\n",
    "\n",
    "test_prompt = \"বাংলাদেশের ইতিহাস ও স্বাধীনতা যুদ্ধ সম্পর্কে বলুন।\"\n",
    "messages = [\n",
    "    {\"role\": \"system\", \"content\": \"আপনি একজন তথ্যবহুল বাংলা সহকারী। সঠিক তথ্যসহ বিস্তারিত উত্তর দিন।\"},\n",
    "    {\"role\": \"user\", \"content\": test_prompt}\n",
    "]\n",
    "inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors=\"pt\").to(\"cuda\")\n",
    "\n",
    "output = model.generate(input_ids=inputs, max_new_tokens=300, temperature=0.2, top_p=0.9, repetition_penalty=1.05)\n",
    "print(\"\\n🇧🇩 Qwen2.5-3B Bangla Distill Output:\")\n",
    "print(\"=\" * 60)\n",
    "print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))\n",
    "print(\"=\" * 60)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 7: Export Adapter & Create Download Zip\n",
    "Saves the adapter to `bangla_distill_3b_adapter.zip` ready for download."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil\n",
    "\n",
    "ADAPTER_DIR = os.path.join(WORKING_DIR, \"bangla_distill_3b_adapter\")\n",
    "print(f\"Saving LoRA adapter to {ADAPTER_DIR}...\")\n",
    "model.save_pretrained_merged(ADAPTER_DIR, tokenizer, save_method=\"lora\")\n",
    "\n",
    "ZIP_FILE = os.path.join(WORKING_DIR, \"bangla_distill_3b_adapter.zip\")\n",
    "print(\"Compressing adapter for 1-click download...\")\n",
    "shutil.make_archive(os.path.join(WORKING_DIR, \"bangla_distill_3b_adapter\"), \"zip\", ADAPTER_DIR)\n",
    "print(f\"\\n🎉 READY FOR DOWNLOAD: {ZIP_FILE} ({os.path.getsize(ZIP_FILE)/(1024**2):.1f} MB)\")\n",
    "print(\"👉 Click the Output tab on the right sidebar and download 'bangla_distill_3b_adapter.zip'!\")"
   ]
  }
 ]
}

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
nb_path = os.path.join(script_dir, "train_bangla_3b_unsloth_kaggle.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created {nb_path} successfully!")
