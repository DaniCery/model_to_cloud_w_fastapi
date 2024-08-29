import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Define the parameter grid for XGBoost
    parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Initialize the XGBoost classifier
    xgb_clf = xgb.XGBClassifier()

    # Set up the RandomizedSearchCV
    logging.info("Initializing RandomizedSearchCV...")
    clf = RandomizedSearchCV(
        xgb_clf,
        parameters,
        verbose=3,     # Increase verbosity to get detailed output
        n_iter=10,     # Number of parameter settings sampled
        n_jobs=-1,     # Use all available cores for computation
        random_state=42, # Optional: For reproducibility
        scoring='f1_weighted' # Use a suitable metric, e.g., 'f1_weighted' for imbalanced data
    )

    logging.info("Fitting model with RandomizedSearchCV...")
    model = clf.fit(X_train, y_train)

    logging.info("Model training completed.")

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def slice_data_computation(test, preds, lb, cat_features, label = 'salary'):
    """
    Function that outputs the performance of the model on slices of the data.

    Inputs
    ------
    X_test : np.array
        Known labels, binarized.
    y_test : np.array
        Known labels, binarized.
    model : pkl
        Model to be inferenced.
    cat_features : list
        List of categorical features to be sliced.
    Returns
    -------
    sliced_data_measures : Dict[List[float]]
    """
    sliced_data_measures = {}
    
    #reset indexes
    test.reset_index(drop=True, inplace=True)

    y = test[label]
    y = lb.transform(y.values).ravel()
    X = test.drop([label], axis=1)

    # transform into series to perform slice
    preds = pd.Series(preds)
    y = pd.Series(y)

    for cat in cat_features:
        # Access the unique values of the categorical feature
        unique_values = X[cat].unique()
        
        for c in unique_values:
            # Filter the test set based on the current categorical value
            subset_X = X[X[cat] == c]
            y_subset = y.loc[subset_X.index]  # Ensure y is indexed correctly

            # Predictions hinerited from test results
            preds_subset = preds.loc[subset_X.index]  # Ensure preds is indexed correctly
            
            # Compute metrics
            precision, recall, fbeta = compute_model_metrics(y_subset, preds_subset)
            sliced_data_measures[c] = [precision, recall, fbeta]
    
    return sliced_data_measures


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds