import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the CSV file
@st.cache
def load_data():
    df = pd.read_csv("your_csv_file.csv", delimiter="\t")
    return df

df = load_data()

# Separate features (offense) and labels (section and punishment)
X = df['Offense']
y_section = df['Section']
y_punishment = df['Punishment']

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
X_vectorized = vectorizer.fit_transform(X)

# Train SVM classifiers
svm_section = SVC(kernel='linear')
svm_section.fit(X_vectorized, y_section)

svm_punishment = SVC(kernel='linear')
svm_punishment.fit(X_vectorized, y_punishment)

# Streamlit app
st.title("Offense Section and Punishment Predictor")

# Input text box for entering the offense
offense_input = st.text_input("Enter the offense:")

# Function to predict section and punishment
def predict(offense):
    text_vectorized = vectorizer.transform([offense])
    predicted_section = svm_section.predict(text_vectorized)[0]
    predicted_punishment = svm_punishment.predict(text_vectorized)[0]
    return predicted_section, predicted_punishment

# Prediction
if st.button("Predict"):
    if offense_input:
        predicted_section, predicted_punishment = predict(offense_input)
        st.write("Predicted Section:", predicted_section)
        st.write("Predicted Punishment:", predicted_punishment)
    else:
        st.warning("Please enter an offense.")
