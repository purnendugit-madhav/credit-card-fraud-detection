# 🛡️ Credit Card Fraud Detection

## 📌 Overview

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using a Random Forest Classifier.

## 🎯 Objective

The objective of this project is to identify potentially fraudulent transactions while dealing with the highly imbalanced nature of credit card transaction data.

## 🧠 Machine Learning Approach

1. Data preprocessing
2. Exploratory Data Analysis
3. Feature engineering
4. Train-test split
5. SMOTE for handling class imbalance
6. Feature scaling using StandardScaler
7. Random Forest classification
8. Model evaluation
9. Streamlit deployment

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Precision | 92.31% |
| Recall | 75.79% |
| F1 Score | 83.24% |
| ROC-AUC | 95.85% |

## 🖥️ Application Features

- Transaction prediction
- Fraud probability
- Risk level
- Legitimate/Fraudulent classification
- Prediction history
- Interactive Streamlit interface

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

```text
Credit_CardFR/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
└── notebooks/
    └── fraud_detection.ipynb
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/purnendugit-madhav/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
cd "/Users/purnendumadhav/Desktop/Credit_CardFR"
python -m streamlit run app.py
```

## 🌐 Live Demo

Check out the live application here: [Credit Card Fraud Detection Streamlit App](https://credit-card-fraud-detection-zudnhezglpbg2pnlzianhn.streamlit.app/)

## 👨‍💻 Author

**Purnendu Madhav**

BCA Data Science
