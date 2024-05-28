import streamlit as st
import torch
import numpy as np
from scipy.special import softmax
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from huggingface_hub import login

# Get Hugging Face token from Streamlit secrets
hf_token = st.secrets["HUGGINGFACE_TOKEN"]

# Authenticate with Hugging Face
login(hf_token)

# Model names on Hugging Face
model_names = {
    "Model 1": "maroua1234/my-distillbert-model",
    #"Model 2": "your-username/your-model-2"
}
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("maroua1234/my-distillbert-model")
model = AutoModelForSequenceClassification.from_pretrained("maroua1234/my-distillbert-model")
# Load models and tokenizers
models = {name: DistilBertForSequenceClassification.from_pretrained(path, use_auth_token=hf_token) for name, path in model_names.items()}
tokenizers = {name: DistilBertTokenizer.from_pretrained(path, use_auth_token=hf_token) for name, path in model_names.items()}

# Label dictionary
label_dict = {'advertising': 0, 'announcement': 1, 'financial information': 2, 'subjective opinion': 3}
label_map = {v: k for k, v in label_dict.items()}

# Streamlit interface
st.title("Tweet Classification")
st.write("Choose a model and classify your tweets:")

# Model selection
model_name = st.selectbox("Choose a model", list(model_names.keys()))

# Input text
tweet = st.text_area("Enter a tweet to classify")

if st.button("Classify"):
    if tweet:
        # Tokenize the tweet
        tokenizer = tokenizers[model_name]
        inputs = tokenizer([tweet], padding=True, truncation=True, return_tensors="pt")

        # Get the model
        model = models[model_name]
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities = softmax(logits.cpu().numpy(), axis=1)[0]

        # Display results
        st.write("Predicted probabilities:")
        for label_id, prob in enumerate(probabilities):
            label_name = label_map[label_id]
            st.write(f"{label_name}: {prob:.4f}")

    else:
        st.write("Please enter a tweet to classify.")
