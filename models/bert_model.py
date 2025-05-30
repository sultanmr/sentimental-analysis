from transformers import TFDistilBertModel, AutoTokenizer
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, Conv1D, Flatten, Reshape
from tensorflow.keras.models import Model
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import numpy as np
from models.base_model import BaseModel

class DistilBERTLayer(Layer):
    def __init__(self, model_name='distilbert-base-uncased', **kwargs):
        super(DistilBERTLayer, self).__init__(**kwargs)
        self.bert = TFDistilBertModel.from_pretrained(model_name)
        self.trainable = True

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state  # Return full sequence output for Conv1D

class CustomBERTModel(BaseModel):

    def __init__(self, model_name="distilbert-base-uncased", max_len=128, 
                 dropout_rate1=0.3, dropout_rate2=0.2, dense_units=64, 
                 learning_rate=0.001, batch_size=16, epochs=3):
        self.model_name = model_name
        self.max_len = max_len
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.build_model()

    def build_model(self):
        # Build custom Keras model
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")
        bert_output = DistilBERTLayer(model_name=self.model_name)([input_ids, attention_mask])
        
        # Add a Reshape layer to ensure correct dimensions for Conv1D
        x = Reshape((self.max_len, bert_output.shape[-1]))(bert_output)
        
        # Conv1D layers
        x = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(x)
        x = Dropout(self.dropout_rate1)(x)
        x = Conv1D(filters=128, kernel_size=2, activation='relu', padding='same')(x)
        x = Dropout(self.dropout_rate2)(x)
        
        # Global Max Pooling instead of Flatten to handle variable length
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = Dense(self.dense_units, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input_ids, attention_mask], outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        return model

    def preprocess(self, texts):
        encodings = self.tokenizer(texts, max_length=self.max_len, padding='max_length',
                              truncation=True, return_tensors='tf')
        return {'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']}



    def fit(self, texts, labels, validation_texts=None, validation_labels=None, validation_split=0.2, **kwargs):
        # Preprocess training data
        encodings = self.preprocess(texts)
        x = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        y = tf.convert_to_tensor(labels)
        
        # Prepare validation data
        validation_data = None
        if validation_texts is not None:
            val_encodings = self.preprocess(validation_texts)
            val_x = {
                'input_ids': val_encodings['input_ids'],
                'attention_mask': val_encodings['attention_mask']
            }
            val_y = tf.convert_to_tensor(validation_labels)
            validation_data = (val_x, val_y)
            validation_split = None  # Disable split if explicit validation data provided
        
        # Train with validation
        self.history = self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            validation_split=validation_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            **kwargs
        ).history
        
        return self.history


    def predict(self, texts):
        encodings = self.preprocess(texts)
        x = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        probs = self.model.predict(x)
        return (probs > 0.5).astype(int).flatten()

    def score(self, texts, labels):
        encodings = self.preprocess(texts)
        x = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        _, accuracy = self.model.evaluate(x, tf.convert_to_tensor(labels), verbose=0)
        return accuracy

def perform_grid_search(X_train, y_train):
    # Create model instance with build_fn as a class
    keras_classifier = KerasClassifier(
        model=CustomBERTModel,
        model_name="distilbert-base-uncased",
        max_len=128,
        verbose=1
    )
    
    # Define the parameter grid
    param_grid = {
        'dropout_rate1': [0.2, 0.3],
        'dropout_rate2': [0.1, 0.2],
        'dense_units': [64, 128],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32],
        'epochs': [3, 5]
    }
    
    # Create GridSearchCV
    grid = GridSearchCV(
        estimator=keras_classifier,
        param_grid=param_grid,
        cv=2,  # Reduced to save time
        n_jobs=1,
        verbose=2,
        scoring='accuracy'
    )
    
    # Perform the grid search
    grid_result = grid.fit(X_train, y_train)
    
    # Print results
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} ({stdev}) with: {param}")
    
    return grid_result