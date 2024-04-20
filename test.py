import streamlit as st
import torch
from transformers import AutoTokenizer, pipeline

# Load model and tokenizer
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model,token="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")
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
