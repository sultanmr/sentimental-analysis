import re
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
import os


class KerasCNN:
    def __init__(self, max_words=5000, max_len=200):
        self.tokenizer = Tokenizer(num_words=max_words)
        self.max_words = max_words
        self.max_len = max_len
        self.model = None
        self.label_encoder = LabelEncoder()

    def preprocess_text(self, texts):
        def clean(text):
            text = text.lower()
            text = re.sub(r"\d+", "", text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub(r"\s+", " ", text).strip()
            return text
        return [clean(text) for text in texts]

    
    def fit(self, X_train, y_train, X_val=None, y_val=None, validation_split=0.2, 
            epochs=20, batch_size=32, grid_search=False, param_grid=None):
        
        # Preprocess training data
        X_train = self.preprocess_text(X_train)
        self.tokenizer.fit_on_texts(X_train)
        self.vocab_size = len(self.tokenizer.word_index) + 2
        sequences = self.tokenizer.texts_to_sequences(X_train)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        y_train_enc = self.label_encoder.fit_transform(y_train)
        epochs = int(os.getenv("EPOCHS", epochs))
        # Preprocess validation data if provided
        validation_data = None
        if X_val is not None:
            X_val = self.preprocess_text(X_val)
            sequences_val = self.tokenizer.texts_to_sequences(X_val)
            padded_val = pad_sequences(sequences_val, maxlen=self.max_len)
            y_val_enc = self.label_encoder.transform(y_val)
            validation_data = (padded_val, y_val_enc)
            validation_split = 0.0

        # Build model wrapper for GridSearch
        def build_model(
            filters_1=64, 
            filters_2=128, 
            dropout_rate=0.5,
            learning_rate=0.001,
            kernel_size=6
        ):
            model = Sequential([
                Input(shape=(self.max_len,)),
                Embedding(input_dim=self.vocab_size, output_dim=100),
                Conv1D(filters=filters_1, kernel_size=kernel_size, activation='relu'),
                Dropout(dropout_rate),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=filters_2, kernel_size=kernel_size, activation='relu'),
                Dropout(dropout_rate),
                Flatten(),
                Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model

        if grid_search and param_grid:
            # GridSearchCV setup
            model = KerasClassifier(
                model=build_model,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Default param grid if none provided
            if param_grid is None:
                param_grid = {
                    'filters_1': [64, 128],
                    'filters_2': [64, 128],
                    'dropout_rate': [0.3, 0.5],
                    'learning_rate': [0.001, 0.0001],
                    'kernel_size': [3, 6]
                }
            
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=1,
                verbose=2,
                scoring='accuracy'
            )
            
            grid_result = grid.fit(padded, y_train_enc)
            
            # Store best model and params
            self.model = grid_result.best_estimator_.model
            self.best_params = grid_result.best_params_
            
            return {
                'best_score': grid_result.best_score_,
                'best_params': grid_result.best_params_,
                'cv_results': grid_result.cv_results_
            }
        else:
            # Normal training without GridSearch
            self.model = build_model()
            history = self.model.fit(
                padded, 
                y_train_enc,
                validation_data=validation_data,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            self.history = history.history
            return self.history


    def predict(self, X_test):
        X_test = self.preprocess_text(X_test)
        sequences = self.tokenizer.texts_to_sequences(X_test)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        preds = self.model.predict(padded)
        return (preds > 0.5).astype(int).flatten()


    def predict_n_proba(self, X_test):
        X_test = self.preprocess_text(X_test)
        sequences = self.tokenizer.texts_to_sequences(X_test)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        
        probs = self.model.predict(padded)  # shape: (n_samples, 1)
        preds = (probs > 0.5).astype(int).flatten()
        
        return preds[0], float(probs[0])  # return first prediction and its confidence
