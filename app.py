import streamlit as st

import numpy as np
import requests
import pickle
import io
from pathlib import Path
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yaml

models = {}

with open('config.yaml') as f:
    config = yaml.safe_load(f)
    models = config["models"]

# Set up page
st.set_page_config(page_title="Model Testing UI", layout="wide")
st.title("🛠️ Model Testing Dashboard")


# Download model from Dropbox
@st.cache_resource(show_spinner=False)
def load_model_from_dropbox(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pickle.load(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Sidebar controls
st.sidebar.header("🔧 Configuration")
selected_model = st.sidebar.selectbox(
    "Select Model", 
    list(models.keys())
)

# Load selected model
model_url = models[selected_model]
with st.spinner(f"Loading {selected_model}..."):
    model = load_model_from_dropbox(model_url)

if not model:
    st.error("Model failed to load. Please check the URL.")
    st.stop()

# Main prediction interface
st.header("📝 Text Analysis")
user_input = st.text_area(
    "Enter your text:", 
    "The product worked great and I'm very satisfied!",
    height=150
)

# Prediction function (customize per model type)
def make_prediction(text, model_name):
    try:
        if hasattr(model, 'predict_proba'):
            # Scikit-learn style models
            proba = model.predict_proba([text])[0]
            pred = model.predict([text])[0]
            print ("predict_proba", pred, proba)
            return pred, proba
        if hasattr(model, 'predict'):            
            if model_name == "Keras_CNN":
                # Preprocess the text exactly like during training
                text_clean = model.preprocess_text([text])
                sequences = model.tokenizer.texts_to_sequences(text_clean)
                padded = pad_sequences(sequences, maxlen=model.max_len)
                return int(pred > 0.5), np.array([1-float(pred), float(pred)])
        
            pred = model.predict([text])[0]
            return int(pred > 0.5), np.array([1-float(pred), float(pred)])
            
           
        #    # Keras/TensorFlow models - needs proper preprocessing
        #     if isinstance(model, KerasCNN):  # Special handling for your KerasCNN
        #         # Preprocess the text exactly like during training
        #         text_clean = model.preprocess_text([text])
        #         sequences = model.tokenizer.texts_to_sequences(text_clean)
        #         padded = pad_sequences(sequences, maxlen=model.max_len)
                
        #         # Get prediction
        #         pred = model.model.predict(padded)[0][0]  # Get single probability
        #         return int(pred > 0.5), float(pred)
        #     else:
        #         # For other Keras models
        #         pred = model.predict([text])[0][0]
        #         return int(pred > 0.5), float(pred)
        else:
            return None, None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Prediction button
if st.button("🚀 Analyze Text"):
    if not user_input.strip():
        st.warning("Please enter some text")
        st.stop()
    
    with st.spinner("Analyzing..."):
        prediction, probabilities = make_prediction(user_input, selected_model)
    
    if prediction is not None:
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔮 Prediction")
            if prediction is not None:
                if prediction == 1:
                    st.success("✅ Positive")
                else:
                    st.error("❌ Negative")
            else:
                st.warning("⚠️ Prediction failed")

        
        
        with col2:
            st.subheader("📊 Confidence")
            if probabilities.any():
                confidence = max(probabilities) * 100
                st.metric(
                    "", 
                    f"{confidence:.1f}%", 
                    delta=f"{confidence-50:.1f}% from neutral"
                )
        
        # Show probability breakdown
        if probabilities.any() and len(probabilities) == 2:
            st.subheader("📈 Probability Distribution")
            prob_df = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability": probabilities
            })
            st.bar_chart(prob_df.set_index("Sentiment"))

# Model info section
st.sidebar.header("ℹ️ Model Information")
st.sidebar.write(f"**Selected Model:** {selected_model}")

# Add some styling
st.markdown("""
<style>
    .stTextArea [data-baseweb=textarea] {
        height: 200px;
    }
    .st-b7 {
        color: white !important;
    }
    .st-cj {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)