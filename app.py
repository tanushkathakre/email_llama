import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("ipc_sections.csv")

# Data preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

data['Processed_Offense'] = data['Offense'].apply(preprocess_text)

# Streamlit app
st.title("IPC Section Prediction and Punishment Recommendation")

# Text input for offense
offense_input = st.text_input("Enter the offense description:")

# Load the text classification pipeline from Hugging Face
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

if offense_input:
    # Prediction
    predicted_section = classifier(offense_input)[0]['label']
    predicted_punishment = data[data['Section'] == int(predicted_section)]['Punishment'].iloc[0]
    
    st.subheader("Predicted Section:")
    st.write(predicted_section)
    
    st.subheader("Recommended Punishment:")
    st.write(predicted_punishment)
