import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
import streamlit as st

# Load CSV data
data = pd.read_csv("ipc_sections.csv", delimiter="\t")

# Preprocess data
X = data['Offense']
y_section = data['Section']
y_punishment = data['Punishment']

# Define pipeline
pipeline_section = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

pipeline_punishment = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Train models
pipeline_section.fit(X, y_section)
pipeline_punishment.fit(X, y_punishment)

# Streamlit app
st.title("Offense Predictor")

offense_input = st.text_input("Enter the offense description:")

if st.button("Predict"):
    predicted_section = pipeline_section.predict([offense_input])[0]
    predicted_punishment = pipeline_punishment.predict([offense_input])[0]
    
    st.write("Predicted Section:", predicted_section)
    st.write("Predicted Punishment:", predicted_punishment)
