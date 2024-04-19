import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

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
    max_similarity = 0
    predicted_section = None
    predicted_punishment = None

    # Tokenize input offense
    input_tokens = word_tokenize(input_offense.lower())

    # Iterate over each row in the data
    for index, row in data.iterrows():
        # Tokenize offense from the dataset
        offense_tokens = word_tokenize(row['Offense'].lower())

        # Calculate similarity between input offense and offense from the dataset
        similarity = nltk.jaccard_distance(set(input_tokens), set(offense_tokens))

        # Update predicted section and punishment if similarity is higher
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_section = row['Section']
            predicted_punishment = row['Punishment']

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
