import streamlit as st
import time
import numpy as np
import requests
import pickle
import io
from pathlib import Path
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yaml
from models.bert_model import CustomBERTModel


# Initialize session state for model caching
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}


models = {}

with open('config.yaml') as f:
    config = yaml.safe_load(f)
    models = config["models"]

# Set up page
st.set_page_config(page_title="Model Testing UI", layout="wide")
st.title("🛠️ Model Testing Dashboard")



def load_model_from_dropbox_old(model_name, url):
    try:        
        response = requests.get(url)
        response.raise_for_status()
        loaded_model = pickle.load(io.BytesIO(response.content))
        return loaded_model

    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def get_from_cache(model_name):
    """Check if model exists in session state cache"""
    return st.session_state.loaded_models.get(model_name)
    

# Download model from Dropbox
@st.cache_resource(show_spinner=False)
def load_model_from_dropbox(model_name, url):
    
    try:
        if 'loaded_models' not in st.session_state:
            st.session_state.loaded_models = {}
        loaded_models = st.session_state.loaded_models

        if model_name in loaded_models:
            return loaded_models[model_name]
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))
        bytes_read = 0
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a bytes buffer to store the downloaded data
        data = io.BytesIO()
        
        for chunk in response.iter_content(chunk_size=8192):
            data.write(chunk)
            bytes_read += len(chunk)
            
            # Update progress bar
            if total_size > 0:
                progress_percent = (bytes_read / total_size)
                progress_bar.progress(min(progress_percent, 1.0))
                status_text.text(f"Downloading {model_name}: {bytes_read/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({progress_percent*100:.1f}%)")
            else:
                progress_bar.progress(0.5)
                status_text.text(f"Downloading {model_name}: {bytes_read/1024/1024:.1f}MB")
        
        # Reset buffer position to the beginning
        data.seek(0)
        
        # Show loading message
        status_text.text(f"Loading {model_name} into memory...")
        
        # Load the model
        loaded_model = pickle.load(data)
        
        # Complete the progress
        progress_bar.progress(1.0)
        status_text.text(f"{model_name} loaded successfully!")
        #time.sleep(0.5)  # Let users see the completion
        #status_text.empty()
        loaded_models[model_name] = loaded_model
        st.session_state.loaded_models = loaded_models

        return loaded_model

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
model = get_from_cache(selected_model)

if model is None:
    with st.spinner(f"Loading {selected_model}..."):
        model = load_model_from_dropbox(selected_model, model_url)

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
        pred, prob = model.predict_n_proba([text])
        pred = int(pred)
        prob = float(prob)
        if pred==1:
            probabilities = np.array([1-prob, prob])
        else:
            probabilities = np.array([prob, 1-prob])
        return pred, probabilities
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