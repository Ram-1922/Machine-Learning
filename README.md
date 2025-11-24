📰 Fake News Detection using Machine Learning (TF-IDF + Logistic Regression)

This project implements a Fake News Classification model using Natural Language Processing (NLP) and Machine Learning. It uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert news text into numerical vectors and applies Logistic Regression to classify whether a given news article is Real or Fake.

The project includes:

Preprocessing and combining datasets

Text cleaning

TF-IDF vectorization

Logistic Regression training

Model evaluation

ROC curve, confusion matrix & classification metrics

Example inline dataset for easy execution

🔍 Features
✔ Inline dataset for quick testing

The script includes a small built-in dataset containing synthetic fake and real news so it can run standalone without external files.

✔ Data preprocessing

Combines title + text into a unified content field

Performs whitespace normalization

Shuffles data with labels

✔ Machine Learning pipeline

TF-IDF Vectorization with n-grams

Logistic Regression Classifier

Complete train-test split

✔ Evaluation Metrics

The model computes:

Precision, Recall, F1-score

Confusion Matrix visualization

ROC Curve

AUC (Area Under Curve)

✔ Visualizations

The code generates:

Class Distribution Bar Chart

Confusion Matrix Heatmap

ROC Curve Plot

All visualizations are created using Matplotlib.

🧠 How It Works

Load dataset
A small synthetic dataset of fake & real news articles is created inside the script.

Clean text
The content is cleaned via regex-based whitespace normalization.

TF-IDF Vectorization
Converts text into numerical form using bigram features.

Model Training
Logistic Regression is trained on the TF-IDF vectors.

Prediction & Evaluation
The model predicts fake vs real labels and generates full evaluation reports.

📊 Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib

Regex (re)

📁 Project Structure
├── fake_news_detection.py     # Main script with training + evaluation
└── README.md                  # Project documentation

🏁 Getting Started
1. Install Dependencies
pip install numpy pandas scikit-learn matplotlib

2. Run the Script
python fake_news_detection.py


The program will automatically:

Train the model

Display performance metrics

Show visualization plots

📌 Future Improvements

Add larger real-world dataset

Deploy as a REST API / Web App

Use advanced NLP models like BERT or LSTMs

Save & load models using joblib
