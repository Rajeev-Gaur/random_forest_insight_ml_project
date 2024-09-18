# Machine Learning Project Using Random Forest

## Overview
This project implements a Random Forest model using a customized dataset. The main components of the project are:

- Data Preprocessing
- Model Training and Evaluation
- Visualization of Results

# Random Forest Classification with SMOTE

## Overview

This project implements a Random Forest classification model to predict outcomes based on a dataset. Due to class imbalance in the dataset, we employed SMOTE (Synthetic Minority Over-sampling Technique) to enhance the predictive performance of the model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. Create a virtual environment (optional):
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
3. Install the required packages:
 pip install -r requirements.txt
Results
After running the model with hyperparameter tuning, the following best parameters were identified:

n_estimators: 400
min_samples_split: 2
min_samples_leaf: 1
max_features: 'log2'
max_depth: None
Model Performance
The evaluation metrics of the model are as follows:

Accuracy: 66.67%
Precision:
Class 0: 0.00
Class 1: 0.67
Recall:
Class 0: 0.00
Class 1: 1.00
F1-Score:
Class 0: 0.00
Class 1: 0.80

Confusion Matrix
[[ 0 10]
 [ 0 20]]

Conclusion
The model shows a high recall for class 1, indicating that it successfully identifies most of the positive cases. However, it fails to predict any instances of class 0, leading to a precision of 0 for that class. This suggests that further tuning or data collection may be necessary to balance the dataset and improve model performance.

Future Work
Explore alternative sampling techniques (e.g., ADASYN, RandomOverSampler).
Experiment with different algorithms (e.g., XGBoost, SVM).
Implement feature selection to improve model accuracy.
License
This project is licensed under the MIT License. 
