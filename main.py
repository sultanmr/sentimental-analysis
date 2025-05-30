
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# Values:
# '0' = all logs (default)
# '1' = filter out INFO logs
# '2' = filter out WARNING logs
# '3' = filter out ERROR logs

import warnings
warnings.filterwarnings('ignore')  # Suppress Python warnings
import tensorflow as tf
import logging
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow internal logs
logging.getLogger('transformers').setLevel(logging.ERROR)  # Suppress Hugging Face logs


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv


from models.keras_cnn import KerasCNN
from models.pytorch_cnn import PyTorchCNN
from models.transformers_model import HFTransformer
from models.traditional import TraditionalModel
from models.bert_model import CustomBERTModel
from models.base_model import BaseModel
from utils import *
from tb_utils import *


# Load and preprocess data
load_dotenv()

csv_url = os.getenv("CSV_PATH")
df = pd.read_csv(csv_url, sep='\t')

X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Liked'], test_size=0.2, random_state=42)

models_dict = {
    "Logistic_Regression": TraditionalModel(),
    #"Keras_CNN": KerasCNN(),
    #"Pytorch_CNN": PyTorchCNN(),
   # "Customize_BERT": CustomBERTModel(),
   # "Huggingface_Transformer": HFTransformer(),
}

accuracy_results = {}
# Enhanced Model Training with TensorBoard
for model_name, model in models_dict.items():
    # Train model
    history = model.fit(X_train.tolist(), y_train.tolist())
    predictions = model.predict(X_test.tolist())
    accuracy = accuracy_score(y_test, predictions)
    accuracy_results[model_name] = accuracy

    # Save model
    save_model(model_name, model)

    # Save to Tensorboard
    writer = setup_tensorboard(model_name)  
    save_history (history, accuracy, writer)  
    save_confusion_matrix(y_test, predictions, model_name, writer)
    if hasattr(model, 'model'):
        try:
            writer.add_graph(model.model, 
                           input_to_model=tf.convert_to_tensor(X_test[:1].tolist()))
        except Exception as e:
            print(f"Couldn't save graph for {model_name}: {e}")
    writer.close()

    


# Print results
print("Model Accuracy Results")
print("="*100)
for model_name, accuracy in accuracy_results.items():
    print(f"{model_name}: {accuracy*100:.2f}%")

