import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
# Load data
x_train_df = pd.read_csv("X_train.csv")
y_train_df = pd.read_csv("y_train.csv")

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize words
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into string
    processed_text = ' '.join(filtered_tokens)
    return processed_text

# Apply preprocessing to 'Facts' column
x_train_df['Facts'] = x_train_df['Facts'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train_df['Facts'])

# Train a classifier
classifier = LogisticRegression()
classifier.fit(x_train_tfidf, y_train_df['winner_index'])

# Streamlit app
st.title("Case Winning Probability Predictor")

# User inputs
petitioner = st.text_input("Petitioner")
respondent = st.text_input("Respondent")
facts = st.text_area("Facts")

# Prediction function
def predict_winning_probability(petitioner, respondent, facts):
    # Preprocess the input facts
    processed_facts = preprocess_text(facts)
    # Vectorize the input using the trained TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([processed_facts])
    # Predict the winning probability
    winning_probability = classifier.predict_proba(input_tfidf)
    return winning_probability

# Predict and display result
if st.button("Predict"):
    if petitioner.strip() == "" or respondent.strip() == "" or facts.strip() == "":
        st.warning("Please enter all inputs.")
    else:
        winning_probability = predict_winning_probability(petitioner, respondent, facts)
        st.success(f"Winning Probability for Petitioner: {winning_probability[0][1]}")
