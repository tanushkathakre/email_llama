import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
     token="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"
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    # cache_dir="/data/yash/base_models",
    # device_map='auto',
    token="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"
)

# Streamlit app
st.title("IPC Section Prediction and Punishment Recommendation")

# Text input for offense
offense_input = st.text_input("Enter the offense description:")

if offense_input:
    # Tokenize input text
    input_ids = tokenizer.encode(offense_input, return_tensors="pt")
    
    # Generate text based on the input
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    
    # Decode the generated text
    predicted_punishment = tokenizer.decode(output[0], skip_special_tokens=True)
    
    st.subheader("Recommended Punishment:")
    st.write(predicted_punishment)
