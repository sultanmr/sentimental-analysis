# Restaurant Review Sentiment Analysis with Ensemble Voting
# Combining Traditional ML, Keras CNN, and PyTorch CNN for enhanced accuracy

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

# Traditional ML imports
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

class RestaurantSentimentAnalyzer:
    def __init__(self, random_seed=42):
        """Initialize the ensemble sentiment analyzer"""
        self.random_seed = random_seed
        self.set_seeds()
        
        # Model placeholders
        self.traditional_model = None
        self.keras_model = None
        self.pytorch_model = None
        
        # Preprocessors
        self.tfidf_vectorizer = None
        self.keras_tokenizer = None
        self.pytorch_tokenizer = None
        
        # Training data info
        self.max_len = None
        self.vocab_size = None
        
    def set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
    def load_data(self, csv_url=None):
        """Load and prepare the dataset"""
        if csv_url is None:
            csv_url = "https://www.dropbox.com/scl/fi/6mvhmvbuyijpt5rwzk12o/Restaurant_Reviews.tsv?rlkey=31dhfnze1subkcsdoa50irtvc&st=77nhe6hr&dl=1"
        
        self.df = pd.read_csv(csv_url, sep='\t')
        self.df['Review_Length'] = self.df['Review'].apply(lambda x: len(x.split()))
        
        print(f"Dataset loaded: {len(self.df)} reviews")
        print(f"Positive reviews: {sum(self.df['Liked'])}")
        print(f"Negative reviews: {len(self.df) - sum(self.df['Liked'])}")
        
        return self.df
    
    def preprocess_text_traditional(self, sentences):
        """Preprocess text for traditional ML models"""
        # Convert to lowercase
        sentences = [sentence.lower() for sentence in sentences]
        
        # Remove punctuation
        sentences = [re.sub(r"[^a-zA-Z\s]", "", sentence) for sentence in sentences]
        
        # Remove extra whitespaces
        sentences = [" ".join(sentence.split()) for sentence in sentences]
        
        # Tokenize
        sentences = [word_tokenize(sentence) for sentence in sentences]
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_sentences = []
        for sentence in sentences:
            filtered_sentence = [word for word in sentence if word not in stop_words]
            filtered_sentences.append(filtered_sentence)
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentences = []
        for sentence in filtered_sentences:
            lemmatized_sentence = [lemmatizer.lemmatize(word) for word in sentence]
            lemmatized_sentences.append(lemmatized_sentence)
        
        return [' '.join(sentence) for sentence in lemmatized_sentences]
    
    def train_traditional_model(self, X_train, y_train):
        """Train traditional ML model with hyperparameter tuning"""
        print("Training Traditional ML Model...")
        
        # Preprocess training data
        X_train_processed = self.preprocess_text_traditional(X_train.tolist())
        
        # Find best configuration
        best_score = -1.0
        best_classifier = None
        best_ngram_size = -1
        
        classifiers = [						
						#LinearSVC(random_state=self.random_seed), 
						LogisticRegression(solver="liblinear", random_state=self.random_seed)]
        ngram_sizes = [1, 2, 3]
        
        for classifier in classifiers:
            for n in ngram_sizes:
                vectorizer = TfidfVectorizer(ngram_range=(1, n), max_features=5000)
                X_train_tfidf = vectorizer.fit_transform(X_train_processed)
                
                f1_scores = cross_val_score(classifier, X_train_tfidf, y_train, cv=5, scoring='f1')
                avg_f1_score = f1_scores.mean()
                
                print(f"Classifier: {type(classifier).__name__}, n-gram: {n} => F1: {avg_f1_score:.3f}")
                
                if avg_f1_score > best_score:
                    best_score = avg_f1_score
                    best_classifier = classifier
                    best_ngram_size = n
        
        # Train final model
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, best_ngram_size), max_features=5000)
        self.traditional_model = Pipeline([
            ('tfidf', self.tfidf_vectorizer),
            ('clf', best_classifier)
        ])
        
        # Fit on original text (pipeline will handle preprocessing)
        X_train_processed = self.preprocess_text_traditional(X_train.tolist())
        self.traditional_model.fit(X_train_processed, y_train)
        
        print(f"Best Traditional Model: {type(best_classifier).__name__} with n-gram={best_ngram_size}, F1={best_score:.3f}")
        
    def create_keras_cnn(self):
        """Create Keras CNN model"""
        model = Sequential([
            Input(shape=(self.max_len,)),
            Embedding(input_dim=self.vocab_size, output_dim=100, name='embedding'),
            Conv1D(filters=64, kernel_size=6, activation='relu', name='conv_1'),
            Dropout(0.5, name='dropout_1'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=6, activation='relu', name='conv_2'),
            Dropout(0.5, name='dropout_2'),
            Flatten(),
            Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def train_keras_model(self, X_train, y_train):
        """Train Keras CNN model"""
        print("Training Keras CNN Model...")
        
        # Prepare tokenizer
        self.keras_tokenizer = Tokenizer(oov_token=True)
        self.keras_tokenizer.fit_on_texts(X_train)
        self.vocab_size = len(self.keras_tokenizer.word_index) + 2
        
        # Encode and pad sequences
        X_train_seq = self.keras_tokenizer.texts_to_sequences(X_train)
        self.max_len = max(len(seq) for seq in X_train_seq)
        X_train_padded = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post')
        
        # Create and train model
        self.keras_model = self.create_keras_cnn()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=3,
            restore_best_weights=True
        )
        
        history = self.keras_model.fit(
            X_train_padded.astype('float32'),
            y_train.astype('int32'),
            epochs=15,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("Keras CNN training completed")
        return history
    
    class TextDataset(Dataset):
        """PyTorch Dataset class"""
        def __init__(self, texts, labels):
            self.texts = torch.LongTensor(texts)
            self.labels = torch.FloatTensor(labels)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]
    
    class TextCNN(nn.Module):
        """PyTorch CNN model"""
        def __init__(self, vocab_size, embed_dim, max_len):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=6)
            self.pool = nn.MaxPool1d(kernel_size=2)
            
            # Calculate the size after convolution and pooling
            conv_output_size = max_len - 6 + 1  # After conv1d
            pool_output_size = conv_output_size // 2  # After maxpool
            
            self.fc = nn.Linear(64 * pool_output_size, 1)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.embedding(x)
            x = x.permute(0, 2, 1)
            x = self.pool(F.relu(self.conv1(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            return torch.sigmoid(self.fc(x))
    
    def train_pytorch_model(self, X_train, y_train):
        """Train PyTorch CNN model"""
        print("Training PyTorch CNN Model...")
        
        # Prepare tokenizer (reuse Keras tokenizer if available)
        if self.keras_tokenizer is None:
            self.pytorch_tokenizer = Tokenizer()
            self.pytorch_tokenizer.fit_on_texts(X_train)
            tokenizer = self.pytorch_tokenizer
        else:
            tokenizer = self.keras_tokenizer
        
        # Encode sequences
        sequences = tokenizer.texts_to_sequences(X_train)
        if self.max_len is None:
            self.max_len = max(len(seq) for seq in sequences)
        
        X_train_padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        # Create dataset and dataloader
        train_dataset = self.TextDataset(X_train_padded, y_train.values)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Create model
        vocab_size = len(tokenizer.word_index) + 2
        self.pytorch_model = self.TextCNN(vocab_size=vocab_size, embed_dim=100, max_len=self.max_len)
        
        # Training setup
        optimizer = torch.optim.Adam(self.pytorch_model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        for epoch in range(10):
            self.pytorch_model.train()
            total_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.pytorch_model(X_batch).squeeze()
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = (y_pred >= 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

            acc = correct / total
            train_losses.append(total_loss)
            train_accuracies.append(acc)
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")
        
        print("PyTorch CNN training completed")
        return train_losses, train_accuracies
    
    def predict_traditional(self, X_test):
        """Predict using traditional ML model"""
        X_test_processed = self.preprocess_text_traditional(X_test.tolist())
        return self.traditional_model.predict_proba(X_test_processed)[:, 1]
    
    def predict_keras(self, X_test):
        """Predict using Keras model"""
        X_test_seq = self.keras_tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post')
        return self.keras_model.predict(X_test_padded.astype('float32')).ravel()
    
    def predict_pytorch(self, X_test):
        """Predict using PyTorch model"""
        tokenizer = self.keras_tokenizer if self.keras_tokenizer else self.pytorch_tokenizer
        sequences = tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        self.pytorch_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.LongTensor(X_test_padded)
            outputs = self.pytorch_model(X_test_tensor).squeeze()
            return outputs.numpy()
    
    def ensemble_predict(self, X_test, voting_method='soft'):
        """Make ensemble predictions using voting"""
        # Get predictions from all models
        pred_traditional = self.predict_traditional(X_test)
        pred_keras = self.predict_keras(X_test)
        pred_pytorch = self.predict_pytorch(X_test)
        
        if voting_method == 'soft':
            # Average the probabilities
            ensemble_probs = (pred_traditional + pred_keras + pred_pytorch) / 3
            ensemble_predictions = (ensemble_probs >= 0.5).astype(int)
        else:  # hard voting
            # Convert probabilities to binary predictions first
            bin_traditional = (pred_traditional >= 0.5).astype(int)
            bin_keras = (pred_keras >= 0.5).astype(int)
            bin_pytorch = (pred_pytorch >= 0.5).astype(int)
            
            # Majority vote
            votes = bin_traditional + bin_keras + bin_pytorch
            ensemble_predictions = (votes >= 2).astype(int)
            ensemble_probs = votes / 3  # For consistency
        
        return ensemble_predictions, ensemble_probs, {
            'traditional': pred_traditional,
            'keras': pred_keras,
            'pytorch': pred_pytorch
        }
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and ensemble"""
        results = {}
        
        print("Evaluating Individual Models...")
        
        # Traditional model
        pred_trad = (self.predict_traditional(X_test) >= 0.5).astype(int)
        results['Traditional ML'] = {
            'accuracy': accuracy_score(y_test, pred_trad),
            'predictions': pred_trad
        }
        
        # Keras model
        pred_keras = (self.predict_keras(X_test) >= 0.5).astype(int)
        results['Keras CNN'] = {
            'accuracy': accuracy_score(y_test, pred_keras),
            'predictions': pred_keras
        }
        
        # PyTorch model
        pred_pytorch = (self.predict_pytorch(X_test) >= 0.5).astype(int)
        results['PyTorch CNN'] = {
            'accuracy': accuracy_score(y_test, pred_pytorch),
            'predictions': pred_pytorch
        }
        
        # Ensemble models
        ensemble_soft, _, _ = self.ensemble_predict(X_test, voting_method='soft')
        ensemble_hard, _, _ = self.ensemble_predict(X_test, voting_method='hard')
        
        results['Ensemble (Soft Voting)'] = {
            'accuracy': accuracy_score(y_test, ensemble_soft),
            'predictions': ensemble_soft
        }
        
        results['Ensemble (Hard Voting)'] = {
            'accuracy': accuracy_score(y_test, ensemble_hard),
            'predictions': ensemble_hard
        }
        
        return results
    
    def display_results(self, results):
        """Display comprehensive results"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        for model_name, metrics in results.items():
            print(f"{model_name:25}: {metrics['accuracy']:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
        
        return best_model[0]
    
    def plot_comparison(self, results):
        """Plot model comparison"""
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def detailed_evaluation(self, X_test, y_test, model_name='Ensemble (Soft Voting)'):
        """Show detailed evaluation for best model"""
        if model_name.startswith('Ensemble'):
            voting_method = 'soft' if 'Soft' in model_name else 'hard'
            predictions, _, _ = self.ensemble_predict(X_test, voting_method=voting_method)
        else:
            results = self.evaluate_models(X_test, y_test)
            predictions = results[model_name]['predictions']
        
        print(f"\nDetailed Evaluation for {model_name}")
        print("="*50)
        print(classification_report(y_test, predictions))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=["Dislike", "Like"], 
                   yticklabels=["Dislike", "Like"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = RestaurantSentimentAnalyzer(random_seed=42)
    
    # Load data
    df = analyzer.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['Review'], df['Liked'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['Liked']
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train all models
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    analyzer.train_traditional_model(X_train, y_train)
    analyzer.train_keras_model(X_train, y_train)
    analyzer.train_pytorch_model(X_train, y_train)
    
    # Evaluate models
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    results = analyzer.evaluate_models(X_test, y_test)
    best_model = analyzer.display_results(results)
    
    # Plot comparison
    analyzer.plot_comparison(results)
    
    # Detailed evaluation
    analyzer.detailed_evaluation(X_test, y_test, best_model)
    
    return analyzer, results

# Run the analysis
if __name__ == "__main__":
    analyzer, results = main()