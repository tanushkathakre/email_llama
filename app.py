import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Load the CSV file into a DataFrame
df = pd.read_csv('ipc_sections.csv', sep='\t')

# Preprocess the data
X = df['Offense']
y_section = df['Section']
y_punishment = df['Punishment']

# Define and train the model
model_section = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear'))
])
model_section.fit(X, y_section)

model_punishment = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear'))
])
model_punishment.fit(X, y_punishment)

# Streamlit app
st.title('Offense Predictor')

# Accept user input
offense_text = st.text_input('Enter the offense text:')

# Predict section and punishment
if st.button('Predict'):
    predicted_section = model_section.predict([offense_text])[0]
    predicted_punishment = model_punishment.predict([offense_text])[0]

    st.write(f'Predicted Section: {predicted_section}')
    st.write(f'Predicted Punishment: {predicted_punishment}')
