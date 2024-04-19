import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize

# Suppress warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')

# Load pre-trained BERT tokenizer and model for sequence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.eval()

# Load the CSV file
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

data_file_path = "ipc_sections.csv"
data = load_data(data_file_path)

def predict_section_and_punishment(input_offense, data):
    # Tokenize input sentence using NLTK
    input_tokens = word_tokenize(input_offense)

    # Convert tokens to a single string
    input_text = " ".join(input_tokens)

    # Tokenize input sentence using BERT tokenizer
    encoded_input = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Run input through BERT model
    with torch.no_grad():
        outputs = model(**encoded_input)

    # Get predicted probabilities
    predicted_probs = torch.sigmoid(outputs.logits)

    # Find the index with the highest probability
    predicted_label = torch.argmax(predicted_probs).item()
    
    # Get corresponding section and punishment
    predicted_section = data.loc[predicted_label, 'Section']
    predicted_punishment = data.loc[predicted_label, 'Punishment']
    
    return predicted_section, predicted_punishment


# Streamlit UI
st.title("Offense Predictor")

input_offense = st.text_input("Enter offense details:")
if st.button("Predict"):
    if input_offense:
        predicted_section, predicted_punishment = predict_section_and_punishment(input_offense, data)
        st.write("Predicted Section:", predicted_section)
        st.write("Predicted Punishment:", predicted_punishment)
    else:
        st.warning("Please enter offense details!")
