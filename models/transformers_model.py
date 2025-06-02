from transformers import pipeline


class HFTransformer:
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=False  # Optional: default is False
        )

    def predict (self, X):
        preds = self.classifier(X)
        return [1 if p['label'] == 'POSITIVE' else 0 for p in preds]

    def predict_proba(self, X):
        preds = self.classifier(X)
        return [p['score'] for p in preds]

    def predict_n_proba (self, X):
        preds = self.classifier(X)
        prob = [p['score'] for p in preds][0]
        pred = [1 if p['label'] == 'POSITIVE' else 0 for p in preds][0]
        return pred, round(prob,4)
    def fit(self, X, y=None):
        return None
