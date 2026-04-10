# Breast Cancer Prediction using Machine Learning

## Overview

This project implements a machine learning pipeline to classify breast
tumors as benign or malignant using diagnostic features. The workflow
includes data preprocessing, exploratory data analysis, model training,
and evaluation.

## Problem Statement

The objective is to build a predictive model that can accurately
classify tumor samples based on input features derived from medical
data.

## Dataset

The project uses the Breast Cancer dataset available through
scikit-learn.

-   Features: radius, texture, perimeter, area, smoothness, compactness,
    concavity, symmetry, fractal dimension, etc.
-   Target:
    -   0: Malignant
    -   1: Benign

## Technologies Used

-   Python
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook

## Workflow

### 1. Data Loading

``` python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

### 2. Data Preprocessing

``` python
import pandas as pd

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
```

### 3. Train-Test Split

``` python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Training

``` python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 5. Evaluation

``` python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Results

The model achieves high classification accuracy (\~95% or higher
depending on parameters and splits).

## How to Run

1.  Install dependencies: pip install numpy pandas matplotlib seaborn
    scikit-learn
2. if you're using jupyter cell: !pip install numpy pandas matplotlib seaborn
    scikit-learn

3.  Run the notebook: jupyter notebook
    Breast_Cancer_Prediction_test.ipynb

## Project Structure

Breast-Cancer-Prediction/ │── Breast_Cancer_Prediction_test.ipynb │──
README.md

## Author

Partha Pratim Sarma
