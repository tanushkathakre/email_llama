import nltk
import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
nltk.download('wordnet')
nltk.download('wordnet_ic')
# Load WordNet and information content corpus
wn.ensure_loaded()
brown_ic = wordnet_ic.ic('ic-brown.dat')

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

def predict_section_and_punishment(input_offense):
    # Tokenize input sentence
    encoded_input = tokenizer.encode_plus(
        input_offense,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Run input through BERT model
    with torch.no_grad():
        outputs = model(**encoded_input)

    # Predicted label
    predicted_label = int(torch.sigmoid(outputs.logits).round().item())
    
    # Get corresponding section and punishment
    predicted_section = data.iloc[predicted_label]['Section']
    predicted_punishment = data.iloc[predicted_label]['Punishment']
    
    return predicted_section, predicted_punishment

def find_similar_offense(input_offense):
    max_similarity = -1
    most_similar_offense = None
    
    # Iterate through offenses in the dataset
    for offense in data['Offense']:
        # Calculate similarity using Wu-Palmer similarity
        similarity = max([input_offense.path_similarity(syn) or 0 for syn in wn.synsets(offense)])
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_offense = offense
            
    return most_similar_offense

# Streamlit UI
st.title("Offense Predictor")

input_offense = st.text_input("Enter offense details:")
if st.button("Predict"):
    if input_offense:
        # Find most similar offense in dataset
        similar_offense = find_similar_offense(input_offense)
        st.write("Similar Offense Found:", similar_offense)
        
        # Predict section and punishment for the similar offense
        predicted_section, predicted_punishment = predict_section_and_punishment(similar_offense)
        st.write("Predicted Section:", predicted_section)
        st.write("Predicted Punishment:", predicted_punishment)
    else:
        st.warning("Please enter offense details!")
