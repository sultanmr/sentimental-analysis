# To view TensorBoard:
# tensorboard --logdir logs/

import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

# TensorBoard Setup
def setup_tensorboard(model_name):
    log_dir = Path("logs") / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer

# Save Confusion Matrix
def save_confusion_matrix(y_true, y_pred, model_name, writer, epoch=0):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save to TensorBoard
    writer.add_figure('Confusion Matrix', fig, epoch)
    plt.close(fig)

def save_history (history, accuracy, writer):    
    # Save metrics to TensorBoard
    if history:  # For models that return history
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
            history['loss'], 
            history['accuracy'],
            history.get('val_loss', []), 
            history.get('val_accuracy', [])
        )):
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', acc, epoch)
            if val_loss:
                writer.add_scalar('Loss/val', val_loss, epoch)
            if val_acc:
                writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # Final evaluation metrics
    writer.add_scalar('Accuracy/final', accuracy, 0)