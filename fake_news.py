import os, re, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

FAKE_PATH = "C:\\Users\\srir4\\Downloads\\fake.csv"
TRUE_PATH = "C:\\Users\\srir4\\Downloads\\true.csv"

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)
fake_df["label"] = 1
true_df["label"] = 0
df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(t):
    if not isinstance(t, str): return ""
    return re.sub(r"\s+", " ", t).strip()

df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

plt.bar(["Real (0)", "Fake (1)"], df["label"].value_counts().sort_index())
plt.title("Class Distribution")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)
logit = LogisticRegression(max_iter=2000)
logit.fit(Xtr, y_train)
proba = logit.predict_proba(Xte)[:,1]
preds = (proba >= 0.5).astype(int)

print(classification_report(y_test, preds, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, proba))

cm = confusion_matrix(y_test, preds)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.xticks([0,1], ["Real","Fake"])
plt.yticks([0,1], ["Real","Fake"])
plt.show()

fpr, tpr, _ = roc_curve(y_test, proba)
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, proba):.4f}")
plt.plot([0,1],[0,1],"--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
