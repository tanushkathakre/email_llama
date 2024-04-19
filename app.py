import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
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

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))

# Streamlit app
st.title("IPC Section Prediction and Punishment Recommendation")

# Text input for offense
offense_input = st.text_input("Enter the offense description:")

if offense_input:
    # Preprocess input text
    processed_text = preprocess_text(offense_input)
    
    # Tokenize input text
    inputs = tokenizer(processed_text, padding=True, truncation=True, return_tensors="pt")
    
    # Make prediction
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_section = label_encoder.inverse_transform([predicted_class])[0]
    predicted_punishment = data[data['Section'] == predicted_section]['Punishment'].iloc[0]
    
    st.subheader("Predicted Section:")
    st.write(predicted_section)
    
    st.subheader("Recommended Punishment:")
    st.write(predicted_punishment)
