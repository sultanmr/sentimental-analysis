import re
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from models.base_model import BaseModel



class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # B x E x T
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

class PyTorchCNN(BaseModel):
    def __init__(self, max_words=5000, max_len=200, embed_dim=128, batch_size=32, epochs=50):
        self.max_words = max_words
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = Tokenizer(num_words=max_words)
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def preprocess_text(self, texts):
        def clean(text):
            text = text.lower()
            text = re.sub(r"\d+", "", text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub(r"\s+", " ", text).strip()
            return text
        return [clean(text) for text in texts]
    def fit(self, X_train, y_train, X_val=None, y_val=None, validation_split=0.2):
        # Preprocess training data
        X_train = self.preprocess_text(X_train)
        self.tokenizer.fit_on_texts(X_train)
        sequences = self.tokenizer.texts_to_sequences(X_train)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        y_train_enc = self.label_encoder.fit_transform(y_train)

        # Handle validation data
        if X_val is None and validation_split:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train_enc, y_val_enc = train_test_split(
                padded, y_train_enc, 
                test_size=validation_split,
                random_state=42
            )
        elif X_val is not None:
            X_val = self.preprocess_text(X_val)
            sequences_val = self.tokenizer.texts_to_sequences(X_val)
            X_val = pad_sequences(sequences_val, maxlen=self.max_len)
            y_val_enc = self.label_encoder.transform(y_val)

        # Convert to tensors
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.long),
            torch.tensor(y_train_enc, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.long),
                torch.tensor(y_val_enc, dtype=torch.long)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Initialize model
        self.model = CNNClassifier(self.max_words, self.embed_dim, num_classes=2).to(self.device)
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()

        # History tracking
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            # Store training metrics
            train_loss = epoch_loss / len(train_loader)
            train_acc = correct / total
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)

            # Validation phase
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                # Store validation metrics
                val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

            # Print progress
            if val_loader:
                print(f'Epoch {epoch+1}/{self.epochs} - '
                    f'Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{self.epochs} - '
                    f'Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        
        self.history = history
        return self.history

    def predict(self, X_test):
        self.model.eval()
        X_test = self.preprocess_text(X_test)
        sequences = self.tokenizer.texts_to_sequences(X_test)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        X_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds
