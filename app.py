import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load CSV file
@st.cache
def load_data(csv_file):
    return pd.read_csv(csv_file)

def train_model(train_df):
    # Remove rows with missing values in the 'Offense' column
    train_df = train_df.dropna(subset=['Offense'])

    # Tokenize input data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = [tokenizer.tokenize(str(sent)) for sent in train_df['Offense']]  # Convert to string

    # Convert tokens to IDs
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Pad input sequences
    MAX_LEN = max(len(x) for x in input_ids)
    input_ids = torch.tensor([i + [0]*(MAX_LEN-len(i)) for i in input_ids])

    # Define labels
    labels = torch.tensor(train_df['Section'].astype('category').cat.codes.values)

    # Create attention mask
    attention_mask = torch.tensor([[1] * len(input_ids[0])])

    # Define dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Load pre-trained BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=train_df['Section'].nunique())

    # Fine-tune BERT model
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        for batch in dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    return model, tokenizer


# Predict section and punishment
def predict(model, tokenizer, train_df, offense):
    tokenized_text = tokenizer.tokenize(offense)
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_text)).unsqueeze(0)
    attention_mask = torch.tensor([[1] * len(input_ids[0])])
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    predicted_label_idx = torch.argmax(outputs[0]).item()
    predicted_section = train_df['Section'].cat.categories[predicted_label_idx]
    predicted_punishment = train_df[train_df['Section'] == predicted_section]['Punishment'].iloc[0]
    return predicted_section, predicted_punishment

# Streamlit UI
st.title("Offense Section and Punishment Predictor")

# Load CSV file
csv_file_path = "path_to_your_dataset.csv"  # Update with the path to your CSV file
train_df = load_data(csv_file_path)

# Train model
st.write("Training BERT model...")
model, tokenizer = train_model(train_df)
st.write("Training complete!")

# Prediction
offense_input = st.text_input("Enter offense details:")
if offense_input:
    predicted_section, predicted_punishment = predict(model, tokenizer, train_df, offense_input)
    st.write("Predicted Section:", predicted_section)
    st.write("Predicted Punishment:", predicted_punishment)
