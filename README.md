# Diabetes Risk Prediction with Machine Learning and MLOps

## Project Overview

This project is an end-to-end machine learning system for predicting diabetes risk using health indicator data. The goal is not only to train machine learning models, but also to build a resume-grade ML project with proper data validation, model comparison, experiment tracking, explainability, and deployment.

Because this is a healthcare-related prediction problem, the project focuses on more than accuracy. Special attention is given to recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix analysis, since missing a high-risk diabetic patient can be more harmful than a false positive prediction.

---

## Project Goals

The main goals of this project are:

* Perform data validation and exploratory data analysis.
* Handle class imbalance in a healthcare dataset.
* Train and compare multiple machine learning models.
* Use advanced gradient boosting models such as XGBoost, LightGBM, and CatBoost.
* Track experiments using MLflow.
* Explain model predictions using SHAP.
* Deploy the trained model using Streamlit and FastAPI.
* Build a clean GitHub-ready ML/MLOps project structure.

---

## Dataset

The dataset used in this project is:

**CDC Diabetes Health Indicators Dataset**

The dataset contains health, lifestyle, demographic, and healthcare-access features collected from survey responses.

### Dataset Summary

| Item                   |                 Value |
| ---------------------- | --------------------: |
| Total Records          |               253,680 |
| Total Columns          |                    22 |
| Input Features         |                    21 |
| Target Column          |     `Diabetes_binary` |
| Problem Type           | Binary Classification |
| Missing Values         |                     0 |
| Duplicate-Looking Rows |                24,206 |

### Target Distribution

| Class | Meaning     |   Count | Percentage |
| ----- | ----------- | ------: | ---------: |
| 0     | No Diabetes | 218,334 |     86.07% |
| 1     | Diabetes    |  35,346 |     13.93% |

The dataset is imbalanced, so accuracy alone is not enough for evaluating model performance.

---

## Features

The dataset includes the following features:

* `HighBP`
* `HighChol`
* `CholCheck`
* `BMI`
* `Smoker`
* `Stroke`
* `HeartDiseaseorAttack`
* `PhysActivity`
* `Fruits`
* `Veggies`
* `HvyAlcoholConsump`
* `AnyHealthcare`
* `NoDocbcCost`
* `GenHlth`
* `MentHlth`
* `PhysHlth`
* `DiffWalk`
* `Sex`
* `Age`
* `Education`
* `Income`

The features include binary, ordinal, and continuous variables.

---

## Day 1 Data Understanding

Initial data validation showed:

* The dataset has 253,680 rows and 22 columns.
* There are no missing values.
* The target variable is highly imbalanced.
* 24,206 duplicate-looking rows were identified.
* Since this is survey-style data with many binary and ordinal features, repeated rows may represent different respondents with identical health profiles. Therefore, duplicates are not removed blindly during the initial analysis.

### Feature Type Summary

| Feature Type                     | Count |
| -------------------------------- | ----: |
| Binary Columns                   |    15 |
| Multi-category / Ordinal Columns |     4 |
| Continuous Columns               |     3 |

### Top Features Correlated with Diabetes

| Feature              | Correlation |
| -------------------- | ----------: |
| GenHlth              |      0.2936 |
| HighBP               |      0.2631 |
| DiffWalk             |      0.2183 |
| BMI                  |      0.2168 |
| HighChol             |      0.2003 |
| Age                  |      0.1774 |
| HeartDiseaseorAttack |      0.1773 |
| PhysHlth             |      0.1713 |
| Income               |      0.1639 |
| Education            |      0.1245 |

Initial analysis shows that general health, high blood pressure, difficulty walking, BMI, high cholesterol, age, and physical health are strongly associated with diabetes status.

---

## Machine Learning Models

The project will compare both baseline and advanced machine learning models.

### Baseline Models

* Logistic Regression
* Random Forest

### Advanced Models

* HistGradientBoostingClassifier
* XGBoost
* LightGBM
* CatBoost

### Optional Advanced Add-ons

* TabNet
* AutoGluon

---

## Evaluation Metrics

Since the dataset is imbalanced, the project will evaluate models using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix

For this healthcare use case, recall and F1-score are especially important because the model should reduce false negatives when identifying diabetic patients.

---

## MLOps Components

This project includes the following MLOps components:

* MLflow for experiment tracking
* SHAP for model explainability
* Joblib for saving and loading trained models
* Streamlit for interactive model demo
* FastAPI for model serving as an API
* Docker for containerization
* GitHub for version control

---

## Project Structure

```text
diabetes-risk-prediction-mlops/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_data_validation_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
├── reports/
│   └── figures/
├── app/
│   ├── streamlit_app.py
│   └── api.py
│
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
└── Dockerfile
```

---

## Folder Explanation

### `data/raw/`

Stores the original downloaded dataset.
Raw data is not modified directly.

### `data/processed/`

Stores cleaned or transformed datasets created during preprocessing.

### `notebooks/`

Stores Jupyter notebooks used for exploration, EDA, and experimentation.

### `src/`

Contains reusable Python code for preprocessing, training, evaluation, and prediction.

### `models/`

Stores trained model files.

### `reports/`

Stores generated reports, metrics, summaries, and analysis outputs.

### `reports/figures/`

Stores generated plots and visualizations.

### `app/`

Contains deployment code for Streamlit and FastAPI.

### `tests/`

Contains test scripts for checking preprocessing, prediction, and model loading.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-risk-prediction-mlops.git
cd diabetes-risk-prediction-mlops
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/01_data_validation_eda.ipynb
```

### Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

### Run FastAPI App

```bash
uvicorn app.api:app --reload
```

---

