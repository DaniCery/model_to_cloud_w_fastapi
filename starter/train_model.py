# Script to train machine learning model.

from pathlib import Path
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from .ml.data import process_data
from .ml.model import train_model, compute_model_metrics, slice_data_computation, inference
import logging

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)

# Global variable
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def model_pipeline():
    
    data = pd.read_csv(Path("./data/clean_data.csv"))

    # Load in the data
    logging.info("Starting data loading...")
    data = pd.read_csv("./data/clean_data.csv")

    # Optional enhancement: use K-fold cross-validation instead of a train-test split.
    logging.info("Starting train-test split...")
    train, test = train_test_split(data, test_size=0.20)

    logging.info("Processing training data...")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    logging.info("Processing test data...")
    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train, optimize, and save a model.
    logging.info("Training model...")
    model = train_model(X_train, y_train)

    logging.info("Making predictions...")
    preds = inference(model, X_test)

    # Metrics calculation
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # Log precision, recall, and F-beta
    logging.info("Precision: %s", precision)
    logging.info("Recall: %s", recall)
    logging.info("Fbeta: %s", fbeta)

    # Compute and log sliced data measures
    logging.info("Perform data slice metrics calculation...")
    sliced_data_measures = slice_data_computation(test, preds, lb, cat_features)
    logging.info("...Done")

    ###########################
    # Save artifacts in model folder
    ###########################
    model_dir = Path('./model')
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save encoder and lb in pickle files
    with open(model_dir / 'encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)
    with open(model_dir / 'lb.pkl', 'wb') as file:
        pickle.dump(lb, file)

    # Save the model in a pickle file
    with open(model_dir / 'model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save general metrics
    metrics = {
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta,
    }

    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Save sliced data metrics
    with open(model_dir / 'sliced_data_measures.json', 'w') as f:
        json.dump(sliced_data_measures, f)

    return X_test, y_test, model

if __name__ == "__main__":
    model_pipeline()
