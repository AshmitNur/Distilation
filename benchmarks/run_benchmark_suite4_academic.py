"""
Bangla Distill — Benchmark Suite 4: Robust Academic Bengali Benchmark
Covers:
  - BnMMLU (Multitask STEM, Humanities, Bangladesh Studies)
  - BLUCK (Bengali Linguistic Understanding & Cultural Knowledge: Bagdhara, Proverbs, Literature)
  - BanglaMATH (Mathematical Reasoning in Bengali Script)
  - Reading Comprehension (Belebele & BanglaQuAD style)

Evaluates:
  - Vanilla Base Model vs. Bangla Distill Student
  - Category-by-Category Accuracy & Exact-Match Performance
  - Orthographic Quality & Linguistic Integrity

Generates:
  - benchmarks/results/suite4_academic_knowledge.json
  - reports/Bangla_Distill_Benchmark_Suite4_Academic_Knowledge.pdf
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

def _resolve_model_dir(name: str) -> str:
    p1 = os.path.join(PROJECT_ROOT, "models", name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(PROJECT_ROOT, name)
    if os.path.exists(p2): return p2
    return p1

_3B = _resolve_model_dir("bangla_distill_3b_adapter")
_UNSLOTH = _resolve_model_dir("bangla_distill_unsloth_adapter")
_V2 = _resolve_model_dir("final-lora-adapter-v2")

if os.path.exists(_3B):
    ADAPTER_PATH = _3B
    BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    USE_4BIT = True
    MODEL_LABEL = "Qwen2.5-3B-Instruct"
elif os.path.exists(_UNSLOTH):
    ADAPTER_PATH = _UNSLOTH
    BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    USE_4BIT = False
    MODEL_LABEL = "Qwen2.5-0.5B-Instruct"
else:
    ADAPTER_PATH = _V2
    BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    USE_4BIT = False
    MODEL_LABEL = "Qwen2.5-0.5B-Instruct"

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SYSTEM_PROMPT = (
    "আপনি একজন বাংলা ভাষায় উচ্চশিক্ষিত বিশেষজ্ঞ সহকারী। "
    "প্রশ্নটি মনোযোগ দিয়ে পড়ুন, সবচেয়ে সঠিক বিকল্পটি (A, B, C, বা D) নির্বাচন করুন এবং সংক্ষেপে নির্ভুল বাংলায় স্পষ্ট ব্যাখ্যা দিন।"
)

# Academic Benchmark Batteries
ACADEMIC_TEST_BATTERY = [
    # ==========================================
    # 1. BnMMLU: STEM & Humanities
    # ==========================================
    {
        "id": "bnmmlu_01",
        "category": "BnMMLU (Computer Science)",
        "question": "কম্পিউটারে বাইনারি পদ্ধতিতে '১০১০' (1010) এর দশমিক (Decimal) মান কত?",
        "options": {
            "A": "৮ (Eight)",
            "B": "১০ (Ten)",
            "C": "১২ (Twelve)",
            "D": "১৪ (Fourteen)"
        },
        "answer": "B",
        "explanation": "বাইনারি ১০১০ = ১×২³ + ০×২² + ১×২¹ + ০×২⁰ = ৮ + ০ + ২ + ০ = ১০।"
    },
    {
        "id": "bnmmlu_02",
        "category": "BnMMLU (Physics)",
        "question": "আলোর প্রতিসরণের ক্ষেত্রে স্নেলের সূত্র (Snell's Law) নিচের কোন গাণিতিক সম্পর্কটি প্রকাশ করে?",
        "options": {
            "A": "আপতন কোণ ও প্রতিফলন কোণ সর্বদা সমান (∠i = ∠r)",
            "B": "আপতন কোণের সাইন ও প্রতিসরণ কোণের সাইনের অনুপাত একটি ধ্রুবক (sin i / sin r = μ)",
            "C": "আলোর বেগ মাধ্যমের প্রতিসরাঙ্কের বর্গের ব্যস্তানুপাতিক",
            "D": "আলো সর্বদা সরলরেখায় চলে"
        },
        "answer": "B",
        "explanation": "স্নেলের সূত্রানুযায়ী নির্দিষ্ট দুটি মাধ্যম ও নির্দিষ্ট রঙের আলোর জন্য sin i / sin r = ধ্রুবক।"
    },
    {
        "id": "bnmmlu_03",
        "category": "BnMMLU (Biology & Chemistry)",
        "question": "উদ্ভিদ দেহে সালোকসংশ্লেষণ (Photosynthesis) প্রক্রিয়ার প্রধান উপজাত (By-product) কোনটি যা বায়ুমণ্ডলে নির্গত হয়?",
        "options": {
            "A": "নাইট্রোজেন গ্যাস",
            "B": "কার্বন ডাই-অক্সাইড গ্যাস",
            "C": "অক্সিজেন গ্যাস",
            "D": "মিথেন গ্যাস"
        },
        "answer": "C",
        "explanation": "সালোকসংশ্লেষণে পানির আলোক বিভাজনের ফলে অক্সিজেন গ্যাস (O₂) উপজাত হিসেবে নির্গত হয়।"
    },
    {
        "id": "bnmmlu_04",
        "category": "BnMMLU (History of Bangladesh)",
        "question": "১৯৭১ সালের মুক্তিযুদ্ধ চলাকালে গঠিত মুজিবনগর সরকারের অস্থায়ী রাষ্ট্রপতি কে ছিলেন?",
        "options": {
            "A": "বঙ্গবন্ধু শেখ মুজিবুর রহমান",
            "B": "সৈয়দ নজরুল ইসলাম",
            "C": "তাজউদ্দীন আহমদ",
            "D": "ক্যাপ্টেন এম মনসুর আলী"
        },
        "answer": "B",
        "explanation": "বঙ্গবন্ধু শেখ মুজিবুর রহমান ছিলেন রাষ্ট্রপতি এবং তাঁর অনুপস্থিতিতে সৈয়দ নজরুল ইসলাম অস্থায়ী রাষ্ট্রপতির দায়িত্ব পালন করেন।"
    },
    {
        "id": "bnmmlu_05",
        "category": "BnMMLU (Geography)",
        "question": "বাংলাদেশের একমাত্র পাহাড়ি দ্বীপ কোনটি?",
        "options": {
            "A": "সেন্ট মার্টিন",
            "B": "মহেশখালী",
            "C": "হাতিয়া",
            "D": "নিঝুম দ্বীপ"
        },
        "answer": "B",
        "explanation": "কক্সবাজার জেলায় অবস্থিত মহেশখালী হলো বাংলাদেশের একমাত্র পাহাড়ি দ্বীপ।"
    },
    {
        "id": "bnmmlu_06",
        "category": "BnMMLU (Data Structures)",
        "question": "কম্পিউটার বিজ্ঞানে কোন ডেটা স্ট্রাকচারটি LIFO (Last In First Out) নীতি মেনে কাজ করে?",
        "options": {
            "A": "কিউ (Queue)",
            "B": "স্ট্যাক (Stack)",
            "C": "অ্যারে (Array)",
            "D": "বাইনারি ট্রি (Binary Tree)"
        },
        "answer": "B",
        "explanation": "স্ট্যাক (Stack) হচ্ছে একটি LIFO ডেটা স্ট্রাকচার, যেখানে সর্বশেষ প্রবেশ করানো উপাদানটি সর্বপ্রথমে বের হয়।"
    },
    {
        "id": "bnmmlu_07",
        "category": "BnMMLU (Bangladesh Constitution)",
        "question": "গণপ্রজাতন্ত্রী বাংলাদেশের সংবিধানের অভিভাবক ও চূড়ান্ত ব্যাখ্যাকারক কোন প্রতিষ্ঠান?",
        "options": {
            "A": "বাংলাদেশ জাতীয় সংসদ সচিবালয়",
            "B": "বাংলাদেশ সুপ্রিম কোর্ট",
            "C": "আইন ও বিচার মন্ত্রণালয়",
            "D": "রাষ্ট্রপতির কার্যালয়"
        },
        "answer": "B",
        "explanation": "সংবিধানের ১০২ ও অন্যান্য অনুচ্ছেদ অনুযায়ী সুপ্রিম কোর্ট সংবিধানের অভিভাবক ও ব্যাখ্যাকারক হিসেবে স্বীকৃত।"
    },
    {
        "id": "bnmmlu_08",
        "category": "BnMMLU (Ecology & Forestry)",
        "question": "বিশ্বের সর্ববৃহৎ ম্যানগ্রোভ বন সুন্দরবনের আনুমানিক কত শতাংশ ভূমি বাংলাদেশ সীমানার মধ্যে অবস্থিত?",
        "options": {
            "A": "প্রায় ৪০%",
            "B": "প্রায় ৫০%",
            "C": "প্রায় ৬০%",
            "D": "প্রায় ৭৫%"
        },
        "answer": "C",
        "explanation": "সুন্দরবনের মোট আয়তনের প্রায় ৬০% বাংলাদেশে এবং বাকি প্রায় ৪০% ভারতের অংশে অবস্থিত।"
    },

    # ==========================================
    # 2. BLUCK: Bengali Linguistic Understanding & Cultural Knowledge
    # ==========================================
    {
        "id": "bluck_01",
        "category": "BLUCK (বাগধারা / Idioms)",
        "question": "বাংলা বাগধারা 'ডুমুরের ফুল' বলতে প্রকৃত অর্থে কী বোঝায়?",
        "options": {
            "A": "অত্যন্ত উজ্জ্বল ও দৃশ্যমান পুষ্প",
            "B": "দুর্লভ বা অদৃশ্য বস্তু",
            "C": "সস্তা বা তুচ্ছ বিষয়",
            "D": "অহংকারী ব্যক্তিত্ব"
        },
        "answer": "B",
        "explanation": "'ডুমুরের ফুল' বাগধারাটির ভাবার্থ হলো দুর্লভ বস্তু বা যা সহজে চোখে পড়ে না।"
    },
    {
        "id": "bluck_02",
        "category": "BLUCK (বাগধারা / Idioms)",
        "question": "'আমড়া কাঠের ঢেঁকি' বাগধারাটির যথাযথ অর্থ নিচের কোনটি?",
        "options": {
            "A": "কাঠের তৈরি প্রাচীন তৈজসপত্র",
            "B": "অপদার্থ বা অকেজো ব্যক্তি",
            "C": "অত্যন্ত কষ্টসহিষ্ণু মানুষ",
            "D": "জ্ঞানী কারিগর"
        },
        "answer": "B",
        "explanation": "'আমড়া কাঠের ঢেঁকি' বাগধারাটি এমন ব্যক্তিকে বোঝায় যার কোনো কাজের উপযোগিতা নেই (অপদার্থ/অকেজো)।"
    },
    {
        "id": "bluck_03",
        "category": "BLUCK (প্রবাদ-প্রবচন / Proverbs)",
        "question": "'নাচতে না জানলে উঠান বাঁকা' প্রবাদটির অন্তর্নিহিত নীতিকথা বা তাৎপর্য কী?",
        "options": {
            "A": "উঠানের অসমতল ভূমি নৃত্যকলার পরিপন্থী",
            "B": "নৃত্যে অনিচ্ছা প্রকাশ করা",
            "C": "নিজের অযোগ্যতার বা ব্যর্থতার দোষ পরিস্থিতি বা পরিবেশের ওপর চাপানো",
            "D": "কঠোর অনুশীলনের অভাব নির্দেশ করা"
        },
        "answer": "C",
        "explanation": "নিজের কাজে পারদর্শিতার অভাব থাকলে পরিস্থিতির ওপর দোষ চাপানোর প্রবণতাকে এই প্রবাদে কটাক্ষ করা হয়।"
    },
    {
        "id": "bluck_04",
        "category": "BLUCK (বাংলা সাহিত্য / Tagore)",
        "question": "রবীন্দ্রনাথ ঠাকুর তাঁর কোন কাব্যগ্রন্থের ইংরেজি রূপান্তরের জন্য ১৯১৩ সালে এশিয়ার প্রথম ব্যক্তি হিসেবে সাহিত্যে নোবেল পুরস্কার লাভ করেন?",
        "options": {
            "A": "সোনার তরী",
            "B": "বলাকা",
            "C": "গীতাঞ্জলি (Song Offerings)",
            "D": "চিত্রা"
        },
        "answer": "C",
        "explanation": "রবীন্দ্রনাথ ঠাকুর ১৯১৩ সালে 'গীতাঞ্জলি' (Song Offerings)-এর কাব্যগুণের জন্য সাহিত্যে নোবেল পুরস্কার লাভ করেন।"
    },
    {
        "id": "bluck_05",
        "category": "BLUCK (বাংলা সাহিত্য / Nazrul)",
        "question": "জাতীয় কবি কাজী নজরুল ইসলামের কালজয়ী 'বিদ্রোহী' কবিতাটি তাঁর কোন বিখ্যাত কাব্যগ্রন্থের অন্তর্গত?",
        "options": {
            "A": "অগ্নিবীণা",
            "B": "বিষের বাঁশী",
            "C": "সাম্যবাদী",
            "D": "প্রলয়শিখা"
        },
        "answer": "A",
        "explanation": "১৯২২ সালে প্রকাশিত নজরুল ইসলামের প্রথম কাব্যগ্রন্থ 'অগ্নিবীণা'-এর দ্বিতীয় কবিতা হলো 'বিদ্রোহী'।"
    },
    {
        "id": "bluck_06",
        "category": "BLUCK (বাংলা সাহিত্য / Modern Poetry)",
        "question": "'বাংলার মুখ আমি দেখিয়াছি, তাই আমি পৃথিবীর রূপ খুঁজিতে যাই না আর' — বিখ্যাত এই পঙ্‌ক্তিটির রচয়িতা কোন কবি?",
        "options": {
            "A": "জসীম উদ্‌দীন",
            "B": "জীবনানন্দ দাশ",
            "C": "সুকান্ত ভট্টাচার্য",
            "D": "শামসুর রাহমান"
        },
        "answer": "B",
        "explanation": "রূপসী বাংলার কবি জীবনানন্দ দাশের 'বাংলার মুখ আমি দেখিয়াছি' কবিতার অমর চরণ এটি।"
    },
    {
        "id": "bluck_07",
        "category": "BLUCK (বাংলা ব্যাকরণ / সন্ধি)",
        "question": "'বিদ্যালয়' শব্দটির সঠিক ব্যাকরণসম্মত সন্ধি বিচ্ছেদ কোনটি?",
        "options": {
            "A": "বিদ্যা + আলয়",
            "B": "বিদ্য + আলয়",
            "C": "বিদ্যা + লয়",
            "D": "বিদ্ + আলয়"
        },
        "answer": "A",
        "explanation": "আ-কার + আ-কার = আ-কার নিয়মে 'বিদ্যা + আলয় = বিদ্যালয়'।"
    },
    {
        "id": "bluck_08",
        "category": "BLUCK (বাংলা ব্যাকরণ / সমাস)",
        "question": "'সিংহ চিহ্নিত আসন = সিংহাসন' — এটি কোন সমাসের যথাযথ উদাহরণ?",
        "options": {
            "A": "দ্বন্দ্ব সমাস",
            "B": "মধ্যপদলোপী কর্মধারয় সমাস",
            "C": "বহুব্রীহি সমাস",
            "D": "তৎপুরুষ সমাস"
        },
        "answer": "B",
        "explanation": "যে কর্মধারয় সমাসে ব্যাসবাক্যের মধ্যস্থিত ব্যাখ্যামূলক পদ (চিহ্নিত) লোপ পায়, তাকে মধ্যপদলোপী কর্মধারয় সমাস বলে।"
    },

    # ==========================================
    # 3. BanglaMATH: Native Mathematical Reasoning
    # ==========================================
    {
        "id": "math_01",
        "category": "BanglaMATH (লাভ-ক্ষতি ও শতকরা)",
        "question": "একটি বই ৪০ টাকায় ক্রয় করে ৫০ টাকায় বিক্রয় করা হলো। ব্যবসায় শতকরা কত লাভ হয়েছে?",
        "options": {
            "A": "২০%",
            "B": "২৫%",
            "C": "৩০%",
            "D": "১৫%"
        },
        "answer": "B",
        "explanation": "লাভ = ৫০ - ৪০ = ১০ টাকা। শতকরা লাভ = (১০ / ৪০) × ১০০% = ২৫%।"
    },
    {
        "id": "math_02",
        "category": "BanglaMATH (বয়স ও অনুপাত)",
        "question": "পিতা ও পুত্রের বর্তমান বয়সের অনুপাত ৫:২। পুত্রের বর্তমান বয়স ১৪ বছর হলে, পিতার বর্তমান বয়স কত বছর?",
        "options": {
            "A": "২৮ বছর",
            "B": "৩৫ বছর",
            "C": "৪০ বছর",
            "D": "৪২ বছর"
        },
        "answer": "B",
        "explanation": "অনুপাতের সাধারণ গুণনীয়ক = ১৪ / ২ = ৭। সুতরাং পিতার বয়স = ৫ × ৭ = ৩৫ বছর।"
    },
    {
        "id": "math_03",
        "category": "BanglaMATH (ঐকিক নিয়ম)",
        "question": "৬ জন লোক একটি পুকুর খননের কাজ ১০ দিনে সম্পন্ন করতে পারে। ওই একই কাজ ৪ দিনে সম্পন্ন করতে কতজন লোকের প্রয়োজন হবে?",
        "options": {
            "A": "১২ জন",
            "B": "১৫ জন",
            "C": "১৮ জন",
            "D": "২০ জন"
        },
        "answer": "B",
        "explanation": "মোট শ্রমদিন = ৬ × ১০ = ৬০ জন-দিন। ৪ দিনে কাজটি করতে লোক লাগবে = ৬০ / ৪ = ১৫ জন।"
    },
    {
        "id": "math_04",
        "category": "BanglaMATH (জ্যামিতি ও পিথাগোরাস)",
        "question": "একটি সমকোণী ত্রিভুজের ভূমি ৪ সেন্টিমিটার এবং লম্ব ৩ সেন্টিমিটার। এর অতিভুজের দৈর্ঘ্য কত সেন্টিমিটার?",
        "options": {
            "A": "৫ সেমি",
            "B": "৬ সেমি",
            "C": "৭ সেমি",
            "D": "৮ সেমি"
        },
        "answer": "A",
        "explanation": "পিথাগোরাসের উপপাদ্য মতে: অতিভুজ = √(৩² + ৪²) = √(৯ + ১৬) = √২৫ = ৫ সেমি।"
    },

    # ==========================================
    # 4. Reading Comprehension (Belebele / BanglaQuAD Style)
    # ==========================================
    {
        "id": "rc_01",
        "category": "Reading Comprehension (মাতৃভাষা দিবস)",
        "passage": "১৯৫২ সালের ২১শে ফেব্রুয়ারি মাতৃভাষা বাংলার মর্যাদার দাবিতে ছাত্র-জনতা বুকের তাজা রক্ত ঢেলে দিয়েছিল। পরবর্তীতে ১৯৯৯ সালের ১৭ নভেম্বর ইউনেস্কো ২১শে ফেব্রুয়ারিকে 'আন্তর্জাতিক মাতৃভাষা দিবস' হিসেবে ঐতিহাসিক স্বীকৃতি প্রদান করে, যা সারা বিশ্বে ভাষাগত বৈচিত্র্য সংরক্ষণের প্রতীক হিসেবে উদযাপিত হয়।",
        "question": "ইউনেস্কো কত সালের কত তারিখে ২১শে ফেব্রুয়ারিকে 'আন্তর্জাতিক মাতৃভাষা দিবস' হিসেবে ঐতিহাসিক স্বীকৃতি প্রদান করেছিল?",
        "options": {
            "A": "১৯৫২ সালের ২১শে ফেব্রুয়ারি",
            "B": "১৯৭১ সালের ২৬শে মার্চ",
            "C": "১৯৯৯ সালের ১৭ নভেম্বর",
            "D": "২০০০ সালের ২১শে ফেব্রুয়ারি"
        },
        "answer": "C",
        "explanation": "অনুচ্ছেদ অনুযায়ী ১৯৯৯ সালের ১৭ নভেম্বর ইউনেস্কো ২১শে ফেব্রুয়ারিকে আন্তর্জাতিক মাতৃভাষা দিবস হিসেবে স্বীকৃতি দেয়।"
    },
    {
        "id": "rc_02",
        "category": "Reading Comprehension (পদ্মা সেতু)",
        "passage": "পদ্মা সেতু বাংলাদেশের একটি বহুমুখী সড়ক-রেল সেতু। এটি মুন্সীগঞ্জের মাওয়া এবং শরীয়তপুরের জাজিরাকে সংযুক্ত করে দেশের দক্ষিণ-পশ্চিমাঞ্চলের ২১টি জেলাকে রাজধানীর সঙ্গে যুক্ত করেছে। মূল সেতুর দৈর্ঘ্য ৬.১৫ কিলোমিটার এবং প্রস্থ ১৮.১০ মিটার। সম্পূর্ণ নিজস্ব অর্থায়নে এই সেতুটি নির্মিত হয়েছে।",
        "question": "পদ্মা সেতুর মূল দৈর্ঘ্য কত কিলোমিটার এবং সেতুটি কোন দুই প্রান্তকে সরাসরি সংযুক্ত করেছে?",
        "options": {
            "A": "৫.৫ কিলোমিটার; ঢাকা ও কুমিল্লা",
            "B": "৬.১৫ কিলোমিটার; মাওয়া ও জাজিরা",
            "C": "৭.২ কিলোমিটার; পাটুরিয়া ও দৌলতদিয়া",
            "D": "৬.৮ কিলোমিটার; রূপসা ও মোংলা"
        },
        "answer": "B",
        "explanation": "অনুচ্ছেদটিতে স্পষ্ট উল্লেখ রয়েছে যে মূল সেতুর দৈর্ঘ্য ৬.১৫ কিলোমিটার এবং এটি মাওয়া ও জাজিরাকে সংযুক্ত করেছে।"
    },
    {
        "id": "rc_03",
        "category": "Reading Comprehension (মহাকাশ ও প্রযুক্তি)",
        "passage": "২০১৮ সালের ১২ মে বাংলাদেশ মহাকাশ বিজ্ঞানে ঐতিহাসিক মাইলফলক অর্জন করে, যখন ফ্লোরিডার কেনেডি স্পেস সেন্টার থেকে স্পেসএক্সের ফ্যালকন ৯ রকেটের মাধ্যমে দেশের প্রথম যোগাযোগ উপগ্রহ 'বঙ্গবন্ধু স্যাটেলাইট-১' সফলভাবে উৎক্ষেপণ করা হয়। এর মধ্য দিয়ে বাংলাদেশ বিশ্বের ৫৭তম স্যাটেলাইট উৎক্ষেপণকারী দেশ হিসেবে মর্যাদা লাভ করে।",
        "question": "বঙ্গবন্ধু স্যাটেলাইট-১ উৎক্ষেপণের মাধ্যমে বাংলাদেশ বিশ্বের কততম স্যাটেলাইট অধিকারী দেশ হিসেবে মর্যাদা অর্জন করে?",
        "options": {
            "A": "৫০তম দেশ",
            "B": "৫৭তম দেশ",
            "C": "৬০তম দেশ",
            "D": "৫২তম দেশ"
        },
        "answer": "B",
        "explanation": "অনুচ্ছেদ অনুযায়ী বাংলাদেশ বিশ্বের ৫৭তম দেশ হিসেবে মহাকাশে নিজস্ব স্যাটেলাইট উৎক্ষেপণ করে।"
    },
    {
        "id": "rc_04",
        "category": "Reading Comprehension (বাংলা লিপি ও বর্ণমালা)",
        "passage": "বাংলা বর্ণমালায় মোট ৫০টি বর্ণ রয়েছে। এর মধ্যে স্বরবর্ণ ১১টি এবং ব্যঞ্জনবর্ণ ৩৯টি। মাত্রাভেদে বর্ণগুলোকে তিন ভাগে ভাগ করা হয়: পূর্ণমাত্রার বর্ণ ৩২টি, অর্ধমাত্রার বর্ণ ৮টি এবং মাত্রাহীন বর্ণ ১০টি।",
        "question": "বাংলা বর্ণমালায় মোট বর্ণের সংখ্যা কত এবং এর মধ্যে মাত্রাহীন বর্ণ কয়টি?",
        "options": {
            "A": "৪৮টি বর্ণ; ৮টি মাত্রাহীন",
            "B": "৫০টি বর্ণ; ১০টি মাত্রাহীন",
            "C": "৫২টি বর্ণ; ১২টি মাত্রাহীন",
            "D": "৪৬টি বর্ণ; ৬টি মাত্রাহীন"
        },
        "answer": "B",
        "explanation": "অনুচ্ছেদ অনুযায়ী বাংলা বর্ণমালায় মোট ৫০টি বর্ণ রয়েছে এবং মাত্রাহীন বর্ণের সংখ্যা ১০টি।"
    }
]

def clean_sentence_end(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] in ("।", "?", "!", ".", "\n", '"', "”", "'", "’", ")"):
        return text
    terminators = ["।", "?", "!", "\n"]
    last_idx = -1
    for term in terminators:
        idx = text.rfind(term)
        if idx > last_idx:
            last_idx = idx
    if last_idx > 0 and (last_idx >= len(text) * 0.35):
        return text[:last_idx + 1].strip()
    return text

def extract_selected_option(response_text: str, options: dict = None) -> str:
    """
    Extracts the multiple choice letter (A, B, C, or D) from model output.
    Uses regex across full text, followed by semantic option matching if needed.
    """
    if not response_text:
        return ""
    
    # Priority 1: Direct declaration like 'উত্তর: B' or 'সঠিক বিকল্পটি হলো B' or 'Option B'
    match = re.search(r'(?:উত্তর|সঠিক উত্তর|বিকল্প|সঠিক বিকল্প|সঠিক বিকল্পটি|সঠিক বিকল্পটি হলো|সঠিক বিকল্পটি হল|বিকল্পটি|Option|Answer)\s*(?:হল|হলো|হচ্ছে|is|:)?\s*[:ঃ\-–—]?\s*(?:\*\*)?\(?([A-D])\)?(?:\*\*)?', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # Priority 2: Leading option letter e.g., 'B)' or '(B)' or 'B.' at the start
    match = re.search(r'^\s*\(?([A-D])\)?[.\s]', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Priority 3: Bold choice e.g., **B**
    match = re.search(r'\*\*([A-D])\*\*', response_text)
    if match:
        return match.group(1).upper()

    # Priority 4: Anywhere explicit option declaration e.g., 'B) '
    match = re.search(r'\b([A-D])\)\s', response_text)
    if match:
        return match.group(1).upper()

    # Priority 5: Semantic keyword matching against options
    if options:
        scores = {}
        for k, opt in options.items():
            clean_opt = re.sub(r'\(.*?\)', '', opt).strip()
            terms = [w.strip() for w in re.split(r'[,;।\s]+', clean_opt) if len(w.strip()) >= 2 and w.strip() not in ['হল', 'হলো', 'হচ্ছে', 'একটি', 'কোনটি', 'থেকে', 'বা', 'ও', 'এর', 'হয়ে', 'করে']]
            if terms:
                matched = [t for t in terms if t in response_text]
                scores[k] = len(matched) / len(terms)
            else:
                scores[k] = 1.0 if clean_opt in response_text else 0.0
        best = max(scores, key=scores.get)
        sorted_s = sorted(scores.values(), reverse=True)
        if scores[best] >= 0.4 and (len(sorted_s) < 2 or sorted_s[0] > sorted_s[1]):
            return best

    # Priority 6: Fallback isolated A, B, C, D in first 100 characters
    first_part = response_text[:100]
    match = re.search(r'\b([A-D])\b', first_part)
    if match:
        return match.group(1).upper()

    return ""

def _load_base_lm():
    if USE_4BIT and DEVICE == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="cuda",
            trust_remote_code=True
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            trust_remote_code=True
        )

def run_suite():
    print("=" * 75)
    print("🇧🇩 BANGLA DISTILL — BENCHMARK SUITE 4: ROBUST ACADEMIC KNOWLEDGE")
    print(f"[*] Benchmarking: BnMMLU + BLUCK + BanglaMATH + Reading Comprehension")
    print(f"[*] Architecture: {MODEL_LABEL} ({'4-bit NF4' if USE_4BIT else 'FP16'})")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH if os.path.exists(ADAPTER_PATH) else BASE_MODEL_NAME, trust_remote_code=True)

    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end not in eos_ids:
            eos_ids.append(im_end)
        eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if eot is not None and eot not in eos_ids:
            eos_ids.append(eot)
    eos_arg = eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None)

    # 1. Evaluate Vanilla Base Model
    print(f"\n[Phase 1/2] Evaluating Vanilla Baseline: {MODEL_LABEL} across {len(ACADEMIC_TEST_BATTERY)} questions...")
    base_model = _load_base_lm()
    base_model.eval()

    base_results = []
    for idx, item in enumerate(ACADEMIC_TEST_BATTERY):
        prompt_parts = []
        if "passage" in item and item["passage"]:
            prompt_parts.append(f"অনুচ্ছেদ:\n{item['passage']}\n")
        prompt_parts.append(f"প্রশ্ন: {item['question']}\n")
        prompt_parts.append("বিকল্পসমূহ:")
        for opt_key in ["A", "B", "C", "D"]:
            prompt_parts.append(f"{opt_key}) {item['options'][opt_key]}")
        prompt_parts.append("\nনির্দেশনা: আপনার উত্তরের প্রথম লাইনে অবশ্যই সরাসরি সঠিক বিকল্পটি লিখুন (যেমন: উত্তর: B)। এর পর সংক্ষেপে ১-২ লাইনে নির্ভুল বাংলায় স্পষ্ট ব্যাখ্যা দিন।")
        full_user_prompt = "\n".join(prompt_parts)

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_user_prompt}
        ]
        prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.2,
                top_p=0.85,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_arg
            )
        elapsed = time.time() - t0

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        cleaned_text = clean_sentence_end(raw_text)
        predicted_opt = extract_selected_option(cleaned_text, item["options"])
        is_correct = (predicted_opt == item["answer"])

        base_results.append({
            "item_id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "predicted_option": predicted_opt,
            "is_correct": is_correct,
            "response_text": cleaned_text,
            "latency_s": round(elapsed, 2)
        })
        status_icon = "✅" if is_correct else "❌"
        print(f"   [{idx+1}/{len(ACADEMIC_TEST_BATTERY)}] {status_icon} {item['category']} | Pred: {predicted_opt or 'None'} (Truth: {item['answer']})")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Evaluate Bangla Distill Student
    print(f"\n[Phase 2/2] Evaluating Proposed: Bangla Distill {MODEL_LABEL} Student across {len(ACADEMIC_TEST_BATTERY)} questions...")
    student_base = _load_base_lm()
    student_model = PeftModel.from_pretrained(student_base, ADAPTER_PATH)
    student_model.eval()

    student_results = []
    for idx, item in enumerate(ACADEMIC_TEST_BATTERY):
        prompt_parts = []
        if "passage" in item and item["passage"]:
            prompt_parts.append(f"অনুচ্ছেদ:\n{item['passage']}\n")
        prompt_parts.append(f"প্রশ্ন: {item['question']}\n")
        prompt_parts.append("বিকল্পসমূহ:")
        for opt_key in ["A", "B", "C", "D"]:
            prompt_parts.append(f"{opt_key}) {item['options'][opt_key]}")
        prompt_parts.append("\nনির্দেশনা: আপনার উত্তরের প্রথম লাইনে অবশ্যই সরাসরি সঠিক বিকল্পটি লিখুন (যেমন: উত্তর: B)। এর পর সংক্ষেপে ১-২ লাইনে নির্ভুল বাংলায় স্পষ্ট ব্যাখ্যা দিন।")
        full_user_prompt = "\n".join(prompt_parts)

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_user_prompt}
        ]
        prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            outputs = student_model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.2,
                top_p=0.85,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_arg
            )
        elapsed = time.time() - t0

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        cleaned_text = clean_sentence_end(raw_text)
        predicted_opt = extract_selected_option(cleaned_text, item["options"])
        is_correct = (predicted_opt == item["answer"])

        student_results.append({
            "item_id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "predicted_option": predicted_opt,
            "is_correct": is_correct,
            "response_text": cleaned_text,
            "latency_s": round(elapsed, 2)
        })
        status_icon = "✅" if is_correct else "❌"
        print(f"   [{idx+1}/{len(ACADEMIC_TEST_BATTERY)}] {status_icon} {item['category']} | Pred: {predicted_opt or 'None'} (Truth: {item['answer']})")

    # 3. Calculate Category & Overall Metrics
    categories = ["BnMMLU", "BLUCK", "BanglaMATH", "Reading Comprehension"]
    cat_stats = {}
    for cat in categories:
        b_cat = [r for r in base_results if cat in r["category"]]
        s_cat = [r for r in student_results if cat in r["category"]]
        b_acc = (sum(1 for r in b_cat if r["is_correct"]) / len(b_cat)) * 100 if b_cat else 0
        s_acc = (sum(1 for r in s_cat if r["is_correct"]) / len(s_cat)) * 100 if s_cat else 0
        cat_stats[cat] = {
            "total_items": len(b_cat),
            "base_correct": sum(1 for r in b_cat if r["is_correct"]),
            "student_correct": sum(1 for r in s_cat if r["is_correct"]),
            "base_acc": round(b_acc, 1),
            "student_acc": round(s_acc, 1),
            "gain": round(s_acc - b_acc, 1)
        }

    total_q = len(ACADEMIC_TEST_BATTERY)
    total_base_correct = sum(1 for r in base_results if r["is_correct"])
    total_student_correct = sum(1 for r in student_results if r["is_correct"])
    overall_base_acc = round((total_base_correct / total_q) * 100, 1)
    overall_student_acc = round((total_student_correct / total_q) * 100, 1)
    overall_gain = round(overall_student_acc - overall_base_acc, 1)

    print("\n" + "=" * 60)
    print("🎯 BENCHMARK SUITE 4 SUMMARY RESULTS:")
    print(f"   Vanilla Baseline Accuracy : {total_base_correct}/{total_q} ({overall_base_acc}%)")
    print(f"   Distilled Student Accuracy: {total_student_correct}/{total_q} ({overall_student_acc}%)")
    print(f"   Net Academic Gain         : +{overall_gain}%")
    print("=" * 60)

    # 4. Save Raw JSON Results
    results_payload = {
        "benchmark_suite": "Suite 4: Academic Bengali Knowledge & Reasoning Benchmark",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_architecture": MODEL_LABEL,
        "total_questions": total_q,
        "overall_summary": {
            "base_correct": total_base_correct,
            "student_correct": total_student_correct,
            "base_accuracy": overall_base_acc,
            "student_accuracy": overall_student_acc,
            "gain": overall_gain
        },
        "category_breakdown": cat_stats,
        "detailed_results": {
            "base": base_results,
            "student": student_results
        }
    }

    json_path = os.path.join(RESULTS_DIR, "suite4_academic_knowledge.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Raw results written to: {json_path}")

    # 5. Build Academic PDF Report
    print("[*] Generating academic PDF evaluation report...")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bangla Distill: Academic Bengali Benchmark Report (BnMMLU, BLUCK, BanglaMATH)</title>
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
      line-height: 1.42;
      font-size: 8pt;
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
      margin: 0 0 3px 0;
      letter-spacing: -0.4px;
    }}
    .title-block .subtitle {{
      font-size: 8pt;
      color: #0284c7;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }}
    .card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 10px;
    }}
    .metric-value {{
      font-size: 15pt;
      font-weight: 800;
      color: #0f172a;
      margin: 3px 0 1px 0;
      font-family: 'JetBrains Mono', monospace;
    }}
    .metric-sub {{
      font-size: 6.8pt;
      color: #16a34a;
      font-weight: 600;
    }}
    .metric-label {{
      font-size: 7pt;
      text-transform: uppercase;
      color: #64748b;
      font-weight: 600;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 14px 0;
      font-size: 7.8pt;
    }}
    th {{
      background: #0f172a;
      color: #ffffff;
      padding: 6px 8px;
      text-align: left;
      font-weight: 600;
      font-size: 7.2pt;
      text-transform: uppercase;
    }}
    td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e2e8f0;
    }}
    .gain-pill {{
      display: inline-block;
      background: #dcfce7;
      color: #166534;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 7pt;
    }}
    .badge-correct {{
      display: inline-block;
      background: #dcfce7;
      color: #15803d;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 6.8pt;
    }}
    .badge-incorrect {{
      display: inline-block;
      background: #fee2e2;
      color: #b91c1c;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 6.8pt;
    }}
    .qa-box {{
      border: 1px solid #e2e8f0;
      border-radius: 5px;
      padding: 8px;
      margin-bottom: 8px;
      page-break-inside: avoid;
    }}
    .qa-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
      font-weight: 700;
      font-size: 7.8pt;
    }}
    .qa-content {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 5px;
      font-size: 7.2pt;
    }}
    .pane-base {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 6px;
    }}
    .pane-student {{
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 4px;
      padding: 6px;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title-block">
      <h1>Bangla Distill — Robust Academic Benchmark</h1>
      <div class="subtitle">Suite 4: BnMMLU • BLUCK Cultural Knowledge • BanglaMATH • Comprehension</div>
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 7pt; color: #64748b; text-align: right;">
      <div><strong>Date:</strong> {results_payload["timestamp"]}</div>
      <div><strong>Model:</strong> {MODEL_LABEL} ({'4-bit NF4' if USE_4BIT else 'FP16'})</div>
      <div><strong>Evaluation Batteries:</strong> 24 Academic Standards</div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card">
      <div class="metric-label">Overall Academic Acc</div>
      <div class="metric-value">{overall_student_acc}%</div>
      <div class="metric-sub">▲ +{overall_gain}% vs Base ({overall_base_acc}%)</div>
    </div>
    <div class="card">
      <div class="metric-label">BLUCK (Culture & Idioms)</div>
      <div class="metric-value">{cat_stats["BLUCK"]["student_acc"]}%</div>
      <div class="metric-sub">▲ +{cat_stats["BLUCK"]["gain"]}% vs Base</div>
    </div>
    <div class="card">
      <div class="metric-label">BnMMLU (STEM/History)</div>
      <div class="metric-value">{cat_stats["BnMMLU"]["student_acc"]}%</div>
      <div class="metric-sub">▲ +{cat_stats["BnMMLU"]["gain"]}% vs Base</div>
    </div>
    <div class="card">
      <div class="metric-label">BanglaMATH Reasoning</div>
      <div class="metric-value">{cat_stats["BanglaMATH"]["student_acc"]}%</div>
      <div class="metric-sub">▲ +{cat_stats["BanglaMATH"]["gain"]}% vs Base</div>
    </div>
  </div>

  <div style="font-weight: 700; font-size: 8.8pt; margin: 6px 0 4px 0;">📊 Academic Subcategory Evaluation Matrix</div>
  <table>
    <thead>
      <tr>
        <th>Benchmark Category</th>
        <th>Test Scope</th>
        <th>Sample Size</th>
        <th>Vanilla Base Accuracy</th>
        <th>Distilled Student Accuracy</th>
        <th>Performance Gain</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>BnMMLU Battery</strong></td>
        <td>STEM (CS, Physics, Biology), 1971 War, Constitution, Geography</td>
        <td>8 Items</td>
        <td>{cat_stats["BnMMLU"]["base_acc"]}% ({cat_stats["BnMMLU"]["base_correct"]}/8)</td>
        <td><strong style="color: #15803d;">{cat_stats["BnMMLU"]["student_acc"]}% ({cat_stats["BnMMLU"]["student_correct"]}/8)</strong></td>
        <td><span class="gain-pill">+{cat_stats["BnMMLU"]["gain"]}%</span></td>
      </tr>
      <tr>
        <td><strong>BLUCK Battery</strong></td>
        <td>Bengali Idioms (বাগধারা), Proverbs (প্রবাদ), Tagore, Nazrul, Grammar</td>
        <td>8 Items</td>
        <td>{cat_stats["BLUCK"]["base_acc"]}% ({cat_stats["BLUCK"]["base_correct"]}/8)</td>
        <td><strong style="color: #15803d;">{cat_stats["BLUCK"]["student_acc"]}% ({cat_stats["BLUCK"]["student_correct"]}/8)</strong></td>
        <td><span class="gain-pill">+{cat_stats["BLUCK"]["gain"]}%</span></td>
      </tr>
      <tr>
        <td><strong>BanglaMATH Battery</strong></td>
        <td>Profit/Loss, Ratio & Age, Unitary Method, Geometry in Bengali</td>
        <td>4 Items</td>
        <td>{cat_stats["BanglaMATH"]["base_acc"]}% ({cat_stats["BanglaMATH"]["base_correct"]}/4)</td>
        <td><strong style="color: #15803d;">{cat_stats["BanglaMATH"]["student_acc"]}% ({cat_stats["BanglaMATH"]["student_correct"]}/4)</strong></td>
        <td><span class="gain-pill">+{cat_stats["BanglaMATH"]["gain"]}%</span></td>
      </tr>
      <tr>
        <td><strong>Reading Comprehension</strong></td>
        <td>Language Movement, Padma Bridge, Satellite-1, Bengali Script (QuAD style)</td>
        <td>4 Items</td>
        <td>{cat_stats["Reading Comprehension"]["base_acc"]}% ({cat_stats["Reading Comprehension"]["base_correct"]}/4)</td>
        <td><strong style="color: #15803d;">{cat_stats["Reading Comprehension"]["student_acc"]}% ({cat_stats["Reading Comprehension"]["student_correct"]}/4)</strong></td>
        <td><span class="gain-pill">+{cat_stats["Reading Comprehension"]["gain"]}%</span></td>
      </tr>
      <tr style="background: #f0fdf4;">
        <td colspan="2"><strong>Comprehensive Academic Benchmark Aggregate</strong></td>
        <td><strong>24 Items</strong></td>
        <td><strong>{overall_base_acc}% ({total_base_correct}/24)</strong></td>
        <td><strong style="color: #15803d; font-size: 8.5pt;">{overall_student_acc}% ({total_student_correct}/24)</strong></td>
        <td><span class="gain-pill" style="font-size: 8pt;">+{overall_gain}%</span></td>
      </tr>
    </tbody>
  </table>

  <div style="font-weight: 700; font-size: 8.8pt; margin: 10px 0 6px 0;">🔬 Qualitative Benchmark Audit: Head-to-Head Answers</div>
"""

    for idx, (b, s) in enumerate(zip(base_results, student_results)):
        q_item = ACADEMIC_TEST_BATTERY[idx]
        b_badge = f'<span class="badge-correct">CORRECT ({b["predicted_option"]})</span>' if b["is_correct"] else f'<span class="badge-incorrect">FAIL ({b["predicted_option"] or "None"})</span>'
        s_badge = f'<span class="badge-correct">CORRECT ({s["predicted_option"]})</span>' if s["is_correct"] else f'<span class="badge-incorrect">FAIL ({s["predicted_option"] or "None"})</span>'

        html += f"""
  <div class="qa-box">
    <div class="qa-header bengali">
      <div>#{idx+1}. [{q_item["category"]}] {q_item["question"]}</div>
      <div>Ground Truth: <strong>{q_item["answer"]}</strong></div>
    </div>
    <div class="qa-content bengali">
      <div class="pane-base">
        <div style="font-weight: 700; margin-bottom: 3px; display: flex; justify-content: space-between;">
          <span>Vanilla {MODEL_LABEL}:</span>
          {b_badge}
        </div>
        <div>{b["response_text"]}</div>
      </div>
      <div class="pane-student">
        <div style="font-weight: 700; margin-bottom: 3px; display: flex; justify-content: space-between; color: #15803d;">
          <span>Distilled {MODEL_LABEL} Student:</span>
          {s_badge}
        </div>
        <div>{s["response_text"]}</div>
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
    pdf_out = os.path.join(reports_dir, "Bangla_Distill_Benchmark_Suite4_Academic_Knowledge.pdf")
    html_to_pdf(html, pdf_out)
    print(f"\n🎉 Suite 4 Complete! PDF Report Generated at:\n   {pdf_out}")
    return pdf_out

if __name__ == "__main__":
    run_suite()
