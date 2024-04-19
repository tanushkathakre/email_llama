def predict_section_and_punishment(input_offense, data):
    input_tokens = word_tokenize(input_offense)
    input_text = " ".join(input_tokens)

    encoded_input = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**encoded_input)

    predicted_probs = torch.sigmoid(outputs.logits)
    predicted_probs = predicted_probs.detach().cpu().numpy().flatten()  # Convert to numpy array

    # Find the index with the highest probability across all rows
    predicted_label = np.argmax(predicted_probs)

    # Get corresponding section and punishment from the row with the highest probability
    predicted_section = data.loc[predicted_label, 'Section']
    predicted_punishment = data.loc[predicted_label, 'Punishment']
    
    return predicted_section, predicted_punishment
