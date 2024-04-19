import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import dump, load

# Step 1: Data Preprocessing
df = pd.read_csv("ipc_sections.csv", delimiter="\t")
X = df["Offense"]
y = df["Section"]  # Assuming you want to predict the Section based on the offense

# Step 2: Model Training
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])
model.fit(X, y)

# Save the trained model
dump(model, 'model.joblib')

# Step 3: Streamlit App Development
st.title("Offense Section and Punishment Predictor")

offense_input = st.text_area("Enter the offense:", "Type your offense here...")

# Step 4: Prediction
if st.button("Predict"):
    model = load('model.joblib')
    predicted_section = model.predict([offense_input])
    st.write(f"Predicted Section: {predicted_section}")

    # You can implement a similar process to predict punishment based on the predicted section
