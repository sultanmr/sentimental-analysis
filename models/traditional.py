import os
import re
import string
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# Text Preprocessing Transformer
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"\d+", "", text)  # remove numbers
        text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean_text(text) for text in X]


class TraditionalModel:
    def __init__(self):
        # Define the pipeline
        self.pipeline = Pipeline([
            ('preprocess', TextPreprocessor()),
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        # Define the grid search parameters
        self.param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__solver': ['liblinear', 'lbfgs']
        }

        self.grid = None

    def fit(self, X_train, y_train):
        # We'll use these scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'neg_log_loss': 'neg_log_loss'
        }
        
        self.grid = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=3,
            n_jobs=-1,
            scoring=scoring,
            refit='accuracy',
            return_train_score=True  # This is crucial to get training metrics
        )
        
        self.grid.fit(X_train, y_train)
        
        # Extract the metrics from CV results
        cv_results = self.grid.cv_results_
        
        # Prepare history dictionary
        history = {
            'params': cv_results['params'],
            'mean_train_accuracy': cv_results['mean_train_accuracy'],
            'mean_train_neg_log_loss': cv_results['mean_train_neg_log_loss'],
            'mean_test_accuracy': cv_results['mean_test_accuracy'],
            'mean_test_neg_log_loss': cv_results['mean_test_neg_log_loss'],
            'best_params': self.grid.best_params_,
            'best_score': self.grid.best_score_
        }
        
        print("Best Parameters:", self.grid.best_params_)
        print("Best CV Score:", self.grid.best_score_)


        train_acc = history['mean_train_accuracy']  
        train_loss = [-x for x in history['mean_train_neg_log_loss']]
        val_acc = history['mean_test_accuracy']  
        val_loss = [-x for x in history['mean_test_neg_log_loss']]

        self.history = {
            "accuracy": train_acc,
            "loss": train_loss,
            "val_accuracy": val_acc,
            "val_loss": val_loss
        }
        return self.history

    def predict(self, X_test):
        return self.grid.predict(X_test)

    def predict_proba(self, X_test):
        return self.grid.predict_proba(X_test)


    def predict_n_proba(self, X_test):
        probs = self.grid.predict_proba(X_test)[0]
        pred = np.argmax(probs)
        pred_prob = max(probs)
        return pred, pred_prob
