import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.utils import shuffle

# STEP 1: Load the cleaned dataset
file_path = r"C:\Users\91628\Desktop\Ai Legal Shield\Datasets\training dataset.xlsx"
df = pd.read_excel(file_path)

# STEP 2: Use only 'Case Description' for feature extraction
df['text'] = df['Case Description'].fillna('')

# STEP 3: Shuffle dataset
df = shuffle(df, random_state=42).reset_index(drop=True)

# STEP 4: Define custom stop words
custom_stopwords = ["court", "judgment", "order", "appeal", "petition", "state", "respondent", "petitioner", "filed"]

# STEP 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)
X_tfidf = vectorizer.fit_transform(df['text'])
y = df['label'].values

# STEP 6: Apply K-Fold Cross Validation
model = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(model, X_tfidf, y, cv=10, scoring='accuracy')

print("\nK-Fold Cross Validation Results:")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i} Accuracy: {score:.4f}")
print(f"Average Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}")

# STEP 7: Train-test split for evaluation and saving outputs
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42, stratify=y
)

# STEP 8: Train final model on training set
model.fit(X_train, y_train)

# STEP 9: Evaluate on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# STEP 10: Print numeric results
print("\nFinal Model Evaluation on Holdout Set:")
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("Accuracy:", accuracy)

# STEP 11: Save model and vectorizer
model_path = r"C:\Users\91628\Desktop\Ai Legal Shield\Model"
os.makedirs(model_path, exist_ok=True)
joblib.dump(model, os.path.join(model_path, "logistic_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_path, "tfidf_vectorizer.pkl"))

# STEP 12: Save evaluation metrics
with open(os.path.join(model_path, "model_evaluation.txt"), "w", encoding="utf-8") as f:
    f.write("Model Evaluation\n")
    f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n\n")
    f.write("Classification Report:\n" + class_report + "\n")
    f.write("Accuracy: " + str(accuracy) + "\n\n")
    f.write("K-Fold Cross Validation Accuracies:\n" + str(cv_scores) + "\n")
    f.write("Average CV Accuracy: " + str(np.mean(cv_scores)) + "\n")
    f.write("Standard Deviation: " + str(np.std(cv_scores)) + "\n")

# STEP 13: Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "AI"], yticklabels=["Real", "AI"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(model_path, "confusion_matrix.png"))
plt.show()

# STEP 14: Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(model_path, "roc_curve.png"))
plt.show()

print("\nModel, vectorizer, cross-validation, and plots saved successfully!")