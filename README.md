# ⚡ T5-LoRA Summarizer: The "TL;DR" Generator

**Turn walls of text into bite-sized summaries instantly.**  
*Powered by Google's T5 & Parameter-Efficient Fine-Tuning.*

---

## 🧐 What is this?

Ever opened a news article and thought, *"I don't have time for this"*?

This project uses **Generative AI** to solve that. It is a **T5-Small** model fine-tuned on the massive **CNN/DailyMail** dataset to generate accurate abstractive summaries.

**The Twist?** We used **LoRA (Low-Rank Adaptation)**. Instead of retraining the whole brain, we just trained a tiny adapter.

* **Original Model:** ~60 Million Parameters  
* **Our Trainable Params:** Only ~294,000 (That's **0.48%**!)

---

## ✨ Features

* **🎯 Smart Summarization:** Understands context and rewrites the core message (Abstractive), not just copy-pasting sentences.
* **🏎️ Lightweight & Fast:** Uses `t5-small` + LoRA, making it small enough to run inference on a standard CPU.
* **🎨 Web Interface:** Comes with a built-in **Gradio** app. Paste text -> Get Summary.
* **🛠️ Tech Stack:** PyTorch, Hugging Face Transformers, PEFT, and Evaluate.

---

## 📊 Performance Stats

Don't let the small size fool you. We achieved solid results with minimal compute resources:

| Metric       | Score   | What it means                              |
|--------------|---------|--------------------------------------------|
| **ROUGE-1**  | **38.52** | Good overlap of key words                   |
| **ROUGE-L**  | **25.66** | Strong sentence structure retention          |
| **Gen Length** | **~65 tokens** | Concise, tweet-style summaries              |

---

## 🛠️ Installation

Get this running on your machine in 2 minutes.

**1. Clone the magic**

```bash
git clone https://github.com/your-username/t5-lora-summarizer.git
cd t5-lora-summarizer
```

**2. Install the fuel**

```bash
pip install transformers datasets accelerate peft rouge_score nltk gradio torch
```

---

## 🚀 Run the App

We've included a `app.py` script that merges the LoRA weights and launches a web UI.

```bash
python app.py
```

👉 **Open your browser at:** `http://127.0.0.1:7860`

*Paste an article, hit **Summarize**, and watch the AI work.*

---

## 🧠 Under the Hood

For the machine learning enthusiasts, here is the secret sauce used in `text-summarizer.ipynb`:

* **Base Model:** `google/t5-small`
* **LoRA Config:**
  * Rank (`r`): 8
  * Alpha (`lora_alpha`): 32
  * Dropout: 0.1
  * Target Modules: `q` (query) and `v` (value) in attention blocks.
* **Optimization:** FP16 Mixed Precision, AdamW optimizer.

---

## 📜 Credits

* Fine-tuned by **[Nehal and Ibrahim]**
* Dataset: **CNN/DailyMail** via Hugging Face
* Library support: **Hugging Face PEFT**

---

⭐ **Star this repo if you found it useful!** ⭐
