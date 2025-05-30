from transformers import pipeline
from models.base_model import BaseModel

class HFTransformer(BaseModel):
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def predict(self, X):
        preds = self.classifier(X)        
        return [1 if p['label'] == 'POSITIVE' else 0 for p in preds]


    def fit(self, X, y=None):
        return None
