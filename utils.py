import os
import pickle

def save_model (model_name, model):
    models_path = os.getenv("MODELS_PATH")
    model_path = os.path.join(models_path, model_name)
    if model_name == "Pytorch CNN":
        torch.save(model.state_dict(), f"{model_path}.pth")
    elif model_name == "Keras CNN":
        model.save(f"{model_path}.h5")
    elif model_name == "Huggingface Transformer":
        model.save_pretrained(model_path)
    else:  # For scikit-learn and custom models
        with open(f"{model_path}.pkl", "wb") as f:
            pickle.dump(model, f)


def ensemble_preds(*args):
    """
    Ensemble predictions from multiple models using majority voting.
    
    Args:
        *args: Variable number of prediction arrays (probabilities between 0-1)
        
    Returns:
        tuple: (ensemble_predictions, ensemble_probs)
        - ensemble_predictions: Majority vote predictions (0 or 1)
        - ensemble_probs: Average probability scores
    """
    if len(args) < 2:
        raise ValueError("At least two prediction arrays are required for ensembling")
    
    # Binarize all predictions at threshold 0.5
    binarized_preds = [(pred >= 0.5).astype(int) for pred in args]
    
    # Sum all binarized predictions
    votes = sum(binarized_preds)
    
    # Majority vote (predict 1 if at least half the models predict 1)
    majority_threshold = len(args) / 2
    ensemble_predictions = (votes >= majority_threshold).astype(int)
    
    # Average probability score (confidence)
    ensemble_probs = sum(args) / len(args)
    
    return ensemble_predictions, ensemble_probs
