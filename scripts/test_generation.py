import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_name = 'Qwen/Qwen2.5-0.5B-Instruct'
tok = AutoTokenizer.from_pretrained('bangla_distill_unsloth_adapter')
base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16, device_map='cuda')
model = PeftModel.from_pretrained(base, 'bangla_distill_unsloth_adapter')
model.eval()

prompt = 'বাংলাদেশের ইতিহাস ও স্বাধীনতা যুদ্ধ সম্পর্কে বলুন।'
messages = [
    {'role': 'system', 'content': 'আপনি একজন বাংলা ভাষায় দক্ষ সহকারী। ব্যবহারকারীর প্রশ্নের উত্তর সঠিক তথ্যসহ বিস্তারিতভাবে বাংলায় দিন।'},
    {'role': 'user', 'content': prompt}
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors='pt').to('cuda')

results = []
for temp, do_sample in [(0.0, False), (0.1, True), (0.3, True), (0.75, True)]:
    if not do_sample:
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.1)
    else:
        out = model.generate(**inputs, max_new_tokens=200, temperature=temp, do_sample=True, repetition_penalty=1.1)
    decoded = tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    results.append(f"=== Temperature {temp} (sample={do_sample}) ===\n{decoded}\n")

with open("test_generation_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("Results written to test_generation_output.txt")
