import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
import os
import torch

# --- 1. Configuration ---
BASE_MODEL_NAME = "t5-small"
LORA_MODEL_PATH = "./t5-summarizer-lora" # Path to your local adapter
TASK_PREFIX = "summarize: "
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 150

# Check for files
if not os.path.exists(LORA_MODEL_PATH):
    print(f"Error: LoRA model not found at {LORA_MODEL_PATH}")
    print("Please make sure the 't5-summarizer-lora' folder is in the same directory as app.py")
    exit()

# --- 2. Load Model & Tokenizer (CPU) ---
print("Loading base model...")
# Load the base T5 model
base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

print(f"Loading LoRA adapters from {LORA_MODEL_PATH}...")
# Load the PeftModel and apply the adapters from your saved directory
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

print("Merging LoRA adapters into the base model...")
# Merge the weights into the base model.
# This creates a single, fast model for inference.
model = model.merge_and_unload()

# Load the tokenizer that was saved with the adapter
tokenizer = T5Tokenizer.from_pretrained(LORA_MODEL_PATH)

# Set up for CPU inference
device = torch.device("cpu")
model.to(device)
model.eval() # Set model to evaluation mode

print("Model loaded successfully on CPU.")

# --- 3. Define the Summarization Function ---
def summarize_text(article_text):
    """
    Takes a long article text and returns a short summary.
    """
    if not article_text or article_text.strip() == "":
        return "Please enter an article to summarize."
        
    print("Summarizing... (This may take a few seconds on CPU)")
    
    # 1. Prepare the input
    prompt = TASK_PREFIX + article_text
    inputs = tokenizer(
        prompt, 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    ).to(device) # Send tensors to CPU

    # 2. Generate the summary
    with torch.no_grad(): # Disable gradient calculation for inference
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,  # Use beam search for better quality
            early_stopping=True,
            no_repeat_ngram_size=2 # Prevents repeating word pairs
        )
    
    # 3. Decode the output
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("Summarization complete.")
    return summary

# --- 4. Create and Launch the Gradio Interface ---
print("Building Gradio interface...")
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        # T5 Summarizer (Fine-tuned with LoRA)
        Enter a long news article and the model will generate a short summary.
        """
    )
    
    article_input = gr.Textbox(
        label="Article Text", 
        placeholder="Paste your full article text here...", 
        lines=15
    )
    
    generate_btn = gr.Button("Summarize", variant="primary")
    
    gr.Markdown("---")
    
    summary_output = gr.Textbox(
        label="Generated Summary", 
        lines=5, 
        interactive=False
    )
    
    generate_btn.click(
        fn=summarize_text, 
        inputs=article_input, 
        outputs=summary_output
    )

print("Launching Gradio app... Open the local URL in your browser.")
demo.launch()