import os
import streamlit as st
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

MODEL_DIR = "llama_model"  # Directory to save the model

# Load model and tokenizer
def load_model(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        # Download model and tokenizer if not already downloaded
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.save_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model.save_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model(MODEL_DIR)
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def generate_summary(prompt):
    # Generate summary
    sequences = summarizer(prompt, max_length=150, num_return_sequences=1, top_k=10)
    summary = sequences[0]['generated_text'].split("[/INST]")[1].strip()
    return summary

def main():
    st.title("Text Summarization App")

    # User input
    prompt = st.text_area("Enter text to summarize:", "")

    # Generate summary on button click
    if st.button("Summarize"):
        if prompt:
            summary = generate_summary(prompt)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
