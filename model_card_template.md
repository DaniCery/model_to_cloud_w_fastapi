# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

### Model Details
- Model Name: Income Prediction Model (xgboost with RandomizedSearchCV)

Model Type: Supervised classification model (xgboost)
Version: 1.0
Framework: xgboost
Training Data: UCI Adult Census dataset, containing demographic information and income labels.
Hyperparameters:
- `n_estimators`: Number of gradient boosted trees. Choice is between [50, 100, 200].
- `max_depth`: Maximum tree depth for base learners. Choice is between [3, 4, 5, 6, 7].
- `learning_rate`: Boosting learning rate (xgb's "eta"). Choice is between [0.01, 0.1, 0.2].
- `subsample`: Subsample ratio of the training instance. Choice is between [0.8, 0.9, 1.0].
- `colsample_bytree`: Subsample ratio of columns when constructing each tree. Choice is between [0.8, 0.9, 1.0].

Hyperparameter Tuning: Performed using RandomizedSearchCV, selecting the best combination (model.best_params_): 
{'subsample': 0.9, 'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
Documentation: (https://github.com/DaniCery/model_to_cloud_w_fastapi)

## Intended Use

Purpose: Predicts whether an individual's income exceeds $50K/year based on demographic data.
Intended Users:
Data scientists for benchmarking income classification models.
Researchers analyzing socioeconomic trends.
Organizations performing demographic data analysis.
Use Cases:
Socioeconomic analysis for policy-making.
Identifying high-income segments in market research.

## Training Data

Training Data

Dataset: The model was trained on the UCI Adult Census dataset, a widely used dataset in machine learning for predicting income levels. It includes 32,561 instances with 15 attributes such as age, workclass, education, marital status, occupation, race, sex, and native country.
Data Preprocessing:
Categorical Features: One-hot encoding was applied and label (salary) is binarized with LabelBinarizer
Handling Missing Data: Any missing values in the dataset were imputed before model training.
Sampling: The dataset is relatively balanced but skewed towards individuals earning less than $50K in a ratio of about 3:1. No additional sampling techniques like oversampling or undersampling were used.

## Evaluation Data

- Data Distribution: The validation set has a similar distribution to the training data, ensuring that the model's performance is reflective of its expected behavior in real-world scenarios.

- Evaluation Methods: Cross-validation was used during hyperparameter tuning to ensure the model generalizes well to unseen data. In Scikit-learn, `RandomizedSearchCV` performs k-fold cross-validation by default, where k=5. This means it splits the data into 5 subsets and then runs the model 5 times, each time using a different subset as the validation set and the remaining subsets as the training set

- Test Set: A hold-out set comprising 20% of the total data was used for final model evaluation.

## Metrics
See below. For more see model folder within the package

### Overall Performance:
- "precision": 0.78,
- "recall": 0.66 
- "fbeta": 0.71

### DataSlices:

#### Sampled slice: Sex:
" Female": [0.7485029940119761, 0.6038647342995169, 0.6684491978609626], " Male": [0.7848324514991182, 0.6737320211960636, 0.725050916496945]
#### Sampled slice Race:
" White": [0.7830820770519263, 0.6721782890007189, 0.723404255319149], " Black": [0.7636363636363637, 0.6176470588235294, 0.6829268292682927], " Asian-Pac-Islander": [0.7317073170731707, 0.5660377358490566, 0.6382978723404255], " Other": [1.0, 0.25, 0.4], " Amer-Indian-Eskimo": [0.7, 0.5833333333333334, 0.6363636363636365]
### Sliced Performance Analysis:
- Data sclices is quite balanced. Relationship (recall varies from 0.23 to 0.70) and Occupation (reacall from 0.26 to 0.85) are somehow unbalanced

## Ethical Considerations

### Impact of Predictions:
- Predictions about income can have significant impacts, especially in contexts like hiring, lending, or insurance. Care should be taken to ensure these predictions are used ethically and do not unfairly disadvantage certain groups.
### Fairness:
- Efforts were made to evaluate the model's performance across different demographic groups (e.g., gender, occupation) to identify potential biases.

## Caveats and Recommendations

### Model Limitations:
- The model is trained on historical data, which may not reflect current or future socioeconomic trends.
### Usage Recommendations:
- Use the model responsibly, particularly in decision-making scenarios that impact individuals' lives.
### Future Work:
- Incorporate more recent data and consider using techniques to reduce bias in the model's predictions.
- Consider implementing fairness-aware machine learning techniques in future iterations to mitigate identified biases.
- Explore alternative machine learning models that might offer better performance or interpretability.