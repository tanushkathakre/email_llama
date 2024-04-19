import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

# Encode labels
label_encoder = LabelEncoder()
data['Section_Encoded'] = label_encoder.fit_transform(data['Section'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['Processed_Offense'], data['Section_Encoded'], test_size=0.2, random_state=42)

# Load Llama tokenizer and model
def load_llm():
    llm_model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GGML")
    llm_tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GGML")
    return llm_model, llm_tokenizer

llm_model, llm_tokenizer = load_llm()

# Streamlit app
st.title("IPC Section Prediction and Punishment Recommendation")

# Text input for offense
offense_input = st.text_input("Enter the offense description:")

if offense_input:
    # Generate text using Llama
    generated_text = llm_tokenizer(offense_input, return_tensors="pt")
    output = llm_model.generate(**generated_text)
    predicted_text = llm_tokenizer.decode(output[0], skip_special_tokens=True)

    st.subheader("Predicted Text:")
    st.write(predicted_text)
