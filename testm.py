
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

import io
import pickle
import numpy as np
from dotenv import load_dotenv
load_dotenv()


from models.keras_cnn import KerasCNN
from models.pytorch_cnn import PyTorchCNN
from models.transformers_model import HFTransformer
from models.traditional import TraditionalModel
from models.bert_model import CustomBERTModel


def load_model(model_name):
    try:       
        model_root = os.getenv("MODELS_PATH")
        model_path = os.path.join(model_root, f"{model_name}.pkl")
        print (model_path)
        with open(model_path, "rb") as f:
            return pickle.load(f)
        return None

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return None

def make_prediction(text, model_name):
    try:
        pred, prob = model.predict_n_proba([text])
        return int(pred), np.array([1-float(prob), float(prob)])
      
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None





text = "The product worked great and I'm very satisfied!"
test = "i dont think so that i like that product a lot"

models = []
predictions = []
models_name = []



models_dict = {
    #"Logistic_Regression": None,
    #"Keras_CNN": None,
    #"Pytorch_CNN": None,
    #"Customize_BERT": None,
    "Huggingface_Transformer": HFTransformer(),
}


for model_name, model in models_dict.items():
    if model is None:
        model = load_model (model_name)
    models_name.append(model_name)
    models.append(model)

for model_name, model in zip (models_name, models):
    try:              
        predictions.append (make_prediction(text, model))
        pred, prob = model.predict_n_proba([text])
        print ("pred", pred)
        print ("prob", prob)
        print ("1-prob", 1-prob)
    except Exception as e:
        print ("Error while prediction", e)



for model_name, prediction in zip(models_name, predictions):
    print(model_name)
    print("=" * len(model_name))
    print("Prediction:", prediction)
    
    print()


