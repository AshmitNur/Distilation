# 🇧🇩 Bangla Distill — Local Evaluation Results

**Execution Time:** 2026-09-08 21:57:54  
**Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (975.9 MB Allocated)  
**Base Model:** Qwen/Qwen2.5-0.5B (FP16)  
**Adapter:** LoRA rank=16, alpha=32  

---

### Test 1: History & National Identity

**Prompt:** `বাংলাদেশের ইতিহাস ও স্বাধীনতা যুদ্ধ সম্পর্কে বলুন।`  
*Translation:* Tell me about the history and Liberation War of Bangladesh.  

| Model Variant | Generation Time | Output Text |
| :--- | :--- | :--- |
| **Vanilla Base (Qwen2.5-0.5B)** | 17.38s | বিষয়শ্রেণী: ফিরস্টেট সফলতা<br>ative verb, past tense and past participle -ed  ; comparative -dented ; superlative -eared<br>ative noun<br>1)  The people of Bangladesh suffered terribly from the war. The people of Bangladesh have suffered terrible by the war.<br>2)  We should not be surprised at such great suffering. We should not be surprised at such great suffering.<br><br>ative adjective<br>1)  They were deeply injured in the battle. They were deeply wounded in the battle.<br>2)  He was greatly disappointed with his performance. He was very disappointed with his performance.<br>ative adverb<br>1)  She was very tired after her work. She was extremely tired after her work.<br>2)  It took us an hour to finish our work. It took us only an hour to finish our work.<br><br>ative pronoun<br>1)  The teacher asked me to report to the principal. The teacher asked me to report to the principal.<br>2)  Please tell me your telephone number. Please tell me your phone number.<br><br>adjective<br>1 |
| **Distilled Student (Bangla LoRA)** | 27.88s | বাংলাদেশের ইতিহাস একটি অন্যতম গোষ্ঠী, যা ১৯৮৩ সালের জন্মগ্রহণ করেছিল। তার উপর নয় এই প্রথম ভাষায় সংস্থাপিত আলফেলুল স্টিফোডিড স্টিফোর (১৯৮৩) শহীদৃষ্টি করেন। তিনটি প্রধান বাংলাদেশী প্রতিষ্ঠানের অধিকারী সঙ্গোপাধ্যায়ী ছিলেন।<br><br>১৯� |

---

### Test 2: Science & Artificial Intelligence

**Prompt:** `কৃত্রিম বুদ্ধিমত্তা (Artificial Intelligence) কী এবং এটি কীভাবে কাজ করে?`  
*Translation:* What is Artificial Intelligence and how does it work?  

| Model Variant | Generation Time | Output Text |
| :--- | :--- | :--- |
| **Vanilla Base (Qwen2.5-0.5B)** | 13.66s | তিনি যেমন সোলালেজন হিটান ছিলেন, তাই আমি ভাষণ প্রশ্ন দেয়।ipmap<br>mite<br>তিনি ফিডাউডে অনেক ঘটনায় বৈশ্বর ও শব্দ হিটান থাকেন।ipmap<br>mite<br>তিনি যেমন চাঠায় মনে করেন, তারপর আমি অসুবিধায় উদাহরণ বহু খুব কম গিয়েছি।ipmap<br>mite<br>তিনি যেমন চাঠায় মনে করেন, তারপর আমি অসুবিধায |
| **Distilled Student (Bangla LoRA)** | 26.49s | কৃত্রিম বুদ্ধিমত্তা হলো সম্পূর্ণ অনুসন্ধান যা আইন, গণিত, সমাজ ও ভাষায় সম্পূর্ণ।  এটি ইসলামীরা এবং সাদা শহীদের জন্যও ব্যবস্থা করেছে। এই কাজে আজ মেশিল্ট, ডি-উইডি এবং আইফোনের বৈচ্ছিল্পের মধ্যে প্রকাশিত খবর, মনোবাদী, চিন্তা এবং সম্পূর্ণ |

---

### Test 3: Geography & Nature

**Prompt:** `সুন্দরবনের গুরুত্ব এবং এর রয়েল বেঙ্গল টাইগার সম্পর্কে বিস্তারিত লিখুন।`  
*Translation:* Write in detail about the importance of Sundarbans and its Royal Bengal Tiger.  

| Model Variant | Generation Time | Output Text |
| :--- | :--- | :--- |
| **Vanilla Base (Qwen2.5-0.5B)** | 13.98s | সুন্দরবনের গুরুত্ব:<br>1.  নিম্ন আকাশের উপর প্রভাব বিষয়টি হিসাবে জানো: যে অনেক সহজ ও অসম্ভব ঘর থেকে প্রয়োজনীয় ধরণ হয়েছে, যে অনেক সময় সম্পূর্ণ ছিলেন না, যে অনেক সময় সম্পূর্ণ ছিলেন না, যে অনেক সময় সম্পূর্ণ ছিলেন না, যে অনেক সময় সম্� |
| **Distilled Student (Bangla LoRA)** | 25.97s | আধুনিক আশা হলো, মহান ভাষায় "অসংখ্যক" উপাদানের অন্ধবিশ্বাস-ভৌগোলতা।  এই চিন্তা ব্যবস্থাপনা করে বাস্তবত জীবনের ঘটনার কাছাকাছি করার প্রতি বিশ্বের জ্ঞান, দৃষ্টি ও আশীচরণের সাথে বিবরণ করা হয়।  <br><br>এছাড়াও, আধুনিক আশা ফলে মহান ভাষা সম্পর্কে বিস্তার |

---

### Test 4: Advice & Health

**Prompt:** `একজন শিক্ষার্থীর সুস্থ ও কর্মক্ষম থাকার জন্য তিনটি প্রয়োজনীয় নিয়ম কী?`  
*Translation:* What are three essential rules for a student to stay healthy and productive?  

| Model Variant | Generation Time | Output Text |
| :--- | :--- | :--- |
| **Vanilla Base (Qwen2.5-0.5B)** | 6.09s | যেমন, এখানে আপনি সবচেয়ে দূষা হতে বলেছেন, একটি উপহালিকা বিচার গণিতে পরিচালনার উদাহরণ।accopted<br><br>accopted<br>accopted<br>accopted |
| **Distilled Student (Bangla LoRA)** | 24.69s | আমি আলোচনা করতে হবে। এই অধ্যায়ে ইংরেজি ভাষায় বহু উপাদানের সুস্থ ও কর্মক্ষমতা ব্যবহার করছি:<br><br>---<br><br>**১. খৃষ্টীয় পদ্ধতি**<br><br>খৃষ্টীয় পদ্ধতি ছিলেন ভাইস্ট পণ্ডিতের গণনা, চিন্তা, শিক্ষা, শাসন, ঘটনা, অভিযান, প্রয়োগ, শৈশৈব্য, সরলতা, মন্দির স্বাদ, এ |

---

