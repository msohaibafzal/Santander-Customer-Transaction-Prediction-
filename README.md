# Santander Customer Transaction Prediction

A machine learning project focused on predicting whether a customer will make a transaction using the Santander Customer Transaction Prediction dataset. The project develops a complete machine learning pipeline covering exploratory data analysis, feature preparation, model development, cross-validation, hyperparameter optimization, prediction generation, and SHAP-based model explainability.

The project focuses on high-dimensional tabular data containing 200 anonymized numerical features and uses **ROC-AUC** as the primary evaluation metric.

---

## 📌 Project Overview

The objective of this project is to build a robust machine learning pipeline for the **Santander Customer Transaction Prediction** Kaggle competition.

The project explores and compares different classification approaches, prepares the dataset for machine learning, optimizes a LightGBM model, and uses SHAP to understand the contribution of individual features to model predictions.

The complete workflow includes:

* Exploratory Data Analysis
* Dataset validation and inspection
* Feature preprocessing
* Logistic Regression baseline
* LightGBM classification
* Stratified cross-validation
* Hyperparameter optimization
* Final model training
* Test-set prediction
* Kaggle submission generation
* SHAP-based model explainability

---

## 🎯 Objectives

The main objectives of the project are:

1. Explore and analyze a high-dimensional tabular dataset containing 200 anonymized numerical features.
2. Develop and compare multiple classification models for predicting binary transaction outcomes.
3. Prepare and preprocess the available features appropriately for different machine learning algorithms.
4. Optimize model hyperparameters to improve ROC-AUC performance.
5. Analyze feature importance and model behavior using SHAP.
6. Generate predictions for the Kaggle test dataset and prepare a submission.

---

## 📊 Dataset

The project uses the **Santander Customer Transaction Prediction** dataset from Kaggle.

Each row represents a customer, while the dataset contains 200 anonymized numerical variables ranging from `var_0` to `var_199`.

The target variable is binary:

* `0` — Customer does not make a transaction
* `1` — Customer makes a transaction

The primary evaluation metric is **ROC-AUC**, making probability-based predictions particularly important.

### Dataset Structure

| Dataset  | Samples | Columns |
| -------- | ------: | ------: |
| Training | 200,000 |     202 |
| Testing  | 200,000 |     201 |

The training data contains:

* `ID_code`
* `target`
* 200 numerical features

The test data contains:

* `ID_code`
* 200 numerical features

The dataset contains approximately **90% negative samples and 10% positive samples**, creating a significant class imbalance.

---

## 🔍 Exploratory Data Analysis

The dataset was investigated before model development to understand its structure and quality.

### Dataset Characteristics

The analysis found:

* 200 numerical features
* All features stored as `float64`
* No missing values
* Approximately 90/10 class distribution
* No constant features
* No near-constant features
* Features that are largely uncorrelated with one another

Feature distributions were also inspected using histograms.

Most features appeared approximately normally distributed, although some displayed skewness and heavier tails.

A correlation heatmap was also used to inspect relationships between selected features.

---

## 🧹 Feature Preparation

Feature preprocessing was handled according to the requirements of each model.

### Logistic Regression

Because Logistic Regression is sensitive to feature scale, the numerical features were standardized using:

`StandardScaler`

The Logistic Regression pipeline therefore consisted of:

**Features → StandardScaler → Logistic Regression**

### LightGBM

LightGBM does not require feature standardization, so the original numerical feature values were used directly.

The model was allowed to learn useful feature splits through its tree-based learning process.

### Feature Selection

No explicit feature-selection algorithm was applied.

All 200 numerical features were retained.

LightGBM was allowed to determine useful feature splits internally, while SHAP was later used to analyze feature contributions.

The following columns were excluded from the training feature matrix:

* `ID_code`
* `target`

For the test set:

* `ID_code` was removed from the feature matrix.

---

## 🤖 Models

Several approaches were considered during the project.

### 1. Logistic Regression

Logistic Regression was implemented as a baseline classification model.

Configuration included:

* All 200 features
* StandardScaler preprocessing
* 5-fold Stratified Cross-Validation

This provided a linear baseline against which the tree-based model could be compared.

### 2. LightGBM

LightGBM was selected as the primary tree-based model.

The model was configured with:

* `n_estimators = 1000`
* `learning_rate = 0.01`
* GPU acceleration
* 5-fold Stratified Cross-Validation

LightGBM was selected because of its efficiency on large tabular datasets and its histogram-based, leaf-wise tree growth strategy.

The project also considered LightGBM as a practical alternative to XGBoost because of its computational efficiency and suitability for large tabular datasets.

### 3. Random Forest

A Random Forest configuration using 200 estimators was considered.

However, training was computationally expensive for the 200,000 × 200 dataset, so the corresponding implementation was commented out rather than used as a primary model.

---

## 🔬 Cross-Validation

To obtain a more reliable estimate of model performance, **5-fold Stratified Cross-Validation** was used.

Stratification was important because the target variable is imbalanced.

The folds preserve approximately the same class distribution between training and validation portions.

The general process was:

1. Split the dataset into five stratified folds.
2. Train the model on four folds.
3. Validate on the remaining fold.
4. Calculate ROC-AUC.
5. Repeat until every fold has been used for validation.
6. Compare the resulting performance.

---

## ⚙️ Hyperparameter Optimization

Hyperparameter tuning was performed for LightGBM using **GridSearchCV** with 3-fold cross-validation.

The following parameters were explored:

| Parameter       | Values     |
| --------------- | ---------- |
| `num_leaves`    | 31, 50     |
| `learning_rate` | 0.01, 0.05 |
| `n_estimators`  | 500, 1000  |

This produced:

**2 × 2 × 2 = 8 parameter combinations**

With 3-fold cross-validation:

**8 × 3 = 24 model runs**

The documented best parameter combination was:

| Parameter       | Best Value |
| --------------- | ---------: |
| `num_leaves`    |         31 |
| `learning_rate` |       0.01 |
| `n_estimators`  |       1000 |

---

## 📈 Results

The documented model comparison produced the following ROC-AUC results:

| Model               |  ROC-AUC |
| ------------------- | -------: |
| Logistic Regression |  ~0.8599 |
| LightGBM Default    |  ~0.8719 |
| LightGBM Tuned      | ~0.89233 |

The tuned LightGBM configuration provided the strongest result among the documented model comparisons.

The final model was subsequently trained using the available training data and used to generate predictions for the Kaggle test dataset.

> Note: The project conclusion also reports an approximately 0.902 LightGBM result, while the detailed model-comparison table reports approximately 0.89233 for tuned LightGBM. These values appear to correspond to different evaluation stages or reporting points in the project material.

---

## 🧠 Model Explainability with SHAP

Model performance alone does not explain why a prediction was produced.

To investigate model behavior, **SHAP (SHapley Additive exPlanations)** was used.

A `TreeExplainer` was applied to the trained LightGBM model.

Due to the size of the dataset, a random sample of **5,000 training instances** was used for the SHAP analysis.

### Important Features

The analysis identified the following features among the most influential:

* `var_81`
* `var_139`
* `var_12`
* `var_53`

The SHAP analysis indicated that higher values of most of these influential features generally pushed predictions toward class `1`, while lower values tended to reduce the predicted probability.

The SHAP distributions were relatively symmetric, which was consistent with the generally low inter-feature correlation observed during exploratory analysis.

---

## 🔄 Machine Learning Pipeline

The overall project workflow can be summarized as:

```
Kaggle Dataset
      ↓
Data Loading
      ↓
Dataset Inspection
      ↓
Exploratory Data Analysis
      ↓
Missing-Value & Feature Analysis
      ↓
Feature Preparation
      ↓
┌───────────────────────┐
│                       │
↓                       ↓
Logistic Regression    LightGBM
│                       │
↓                       ↓
StandardScaler       Tree-Based Learning
│                       │
└───────────┬───────────┘
            ↓
    Stratified Cross-Validation
            ↓
    Hyperparameter Optimization
            ↓
    Best LightGBM Model
            ↓
    SHAP Explainability
            ↓
    Full Training
            ↓
    Test Predictions
            ↓
    Kaggle Submission
```

---

## 🛠️ Technologies & Libraries

### Programming Language

* Python 3.12

### Machine Learning

* scikit-learn
* LightGBM
* SHAP

### Data Processing

* pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Visual Studio Code
* Kaggle Notebooks

### Hardware

* MacBook Air
* NVIDIA Tesla T4 GPU through Kaggle Notebooks

---

## 💻 Environment

The documented project environment includes:

| Component    | Version / Configuration |
| ------------ | ----------------------- |
| Python       | 3.12                    |
| scikit-learn | 1.3                     |
| LightGBM     | 4.0 GPU-enabled         |
| SHAP         | 0.44                    |
| pandas       | Kaggle Notebook version |
| NumPy        | Kaggle Notebook version |
| Matplotlib   | Kaggle Notebook version |
| Seaborn      | Kaggle Notebook version |
| GPU          | NVIDIA Tesla T4         |
| Random Seed  | 42                      |

---

## 🚀 Getting Started

### 1. Clone the Repository

```
git clone https://github.com/msohaibafzal/Santander-Customer-Transaction-Prediction-.git

cd Santander-Customer-Transaction-Prediction-
```

### 2. Install Dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap
```

### 3. Obtain the Dataset

Download the Santander Customer Transaction Prediction dataset from Kaggle and place the required files in the appropriate dataset directory.

The project expects:

```
train.csv
test.csv
```

### 4. Run the Notebook

Open the project notebook using Jupyter Notebook, JupyterLab, VS Code, or Kaggle Notebooks.

The notebook performs the complete workflow from exploratory analysis through model training and prediction generation.

---

## 📁 Project Workflow

The implementation follows a structured machine learning workflow:

### Data Analysis

* Dataset shape inspection
* Data type inspection
* Missing-value analysis
* Target distribution analysis
* Feature distribution analysis
* Correlation analysis

### Data Preparation

* Removal of `ID_code`
* Separation of target variable
* Feature scaling for Logistic Regression
* Preservation of original numerical features for LightGBM

### Model Development

* Logistic Regression baseline
* LightGBM classifier
* Random Forest consideration

### Model Evaluation

* Stratified K-Fold Cross-Validation
* ROC-AUC evaluation
* Model comparison

### Optimization

* GridSearchCV
* LightGBM hyperparameter search
* Selection of best documented parameters

### Explainability

* SHAP TreeExplainer
* Feature importance analysis
* SHAP distribution analysis

### Prediction

* Final LightGBM training
* Test-set probability prediction
* Kaggle submission preparation

---

## 📊 Evaluation Metric

The primary evaluation metric used in this project is **ROC-AUC**.

ROC-AUC measures the model's ability to distinguish between positive and negative classes across different classification thresholds.

This is particularly useful for this project because the dataset contains a significant class imbalance, with approximately 10% positive samples.

---

## 🔗 Kaggle Notebook

The project implementation is also available as a Kaggle Notebook:

https://www.kaggle.com/code/supersohaib/ml-project?scriptVersionId=316335645

---

## 👨‍💻 Author

**Muhammad Sohaib Afzal**

Computer Engineer | Automation & Intelligent Systems | AI/ML/DL

GitHub: https://github.com/msohaibafzal

LinkedIn: https://www.linkedin.com/in/msohaibafzal/

---
