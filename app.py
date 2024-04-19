import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# Feature Engineering
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['Processed_Offense'])
y = data['Section']

# Model Building
model = LogisticRegression()
model.fit(X, y)

# Streamlit app
st.title("IPC Section Prediction and Punishment Recommendation")

# Text input for offense
offense_input = st.text_input("Enter the offense description:")

if offense_input:
    # Prediction
    processed_text = preprocess_text(offense_input)
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    predicted_section = model.predict(vectorized_text)[0]
    predicted_punishment = data[data['Section'] == predicted_section]['Punishment'].iloc[0]
    
    st.subheader("Predicted Section:")
    st.write(predicted_section)
    
    st.subheader("Recommended Punishment:")
    st.write(predicted_punishment)
