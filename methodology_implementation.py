# Appendix C: Python Code
# ============================================================
# Comparative Evaluation of Machine Learning Algorithms
# for Cybersecurity Threat Detection
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

# Replace with your dataset path
data = pd.read_csv("NSL_KDD.csv")

print(data.head())

# ------------------------------------------------------------
# Separate Features and Target
# ------------------------------------------------------------

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ------------------------------------------------------------
# Encode Target Variable if Necessary
# ------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ------------------------------------------------------------
# Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# Initialise Models
# ------------------------------------------------------------

models = {

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Support Vector Machine":
        SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        ),

    "K-Nearest Neighbour":
        KNeighborsClassifier(n_neighbors=5)
}

# ------------------------------------------------------------
# Train Models
# ------------------------------------------------------------

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        predictions,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        predictions,
        average='weighted'
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("\n===============================")
    print(name)
    print("===============================")

    print(classification_report(
        y_test,
        predictions
    ))

# ------------------------------------------------------------
# Results Table
# ------------------------------------------------------------

results_df = pd.DataFrame(

    results,

    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print(results_df)

results_df.to_csv(
    "model_results.csv",
    index=False
)

# ------------------------------------------------------------
# Plot Accuracy Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Algorithm"],
    results_df["Accuracy"]
)

plt.ylabel("Accuracy")

plt.title("Accuracy Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Plot Precision Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Algorithm"],
    results_df["Precision"]
)

plt.ylabel("Precision")

plt.title("Precision Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Plot Recall Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Algorithm"],
    results_df["Recall"]
)

plt.ylabel("Recall")

plt.title("Recall Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Plot F1 Comparison
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Algorithm"],
    results_df["F1 Score"]
)

plt.ylabel("F1 Score")

plt.title("F1 Score Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Confusion Matrices
# ------------------------------------------------------------

for name, model in models.items():

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(name)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()

# ------------------------------------------------------------
# ROC Curves
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.3f})"
    )

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.show()

# ------------------------------------------------------------
# Ranking Models
# ------------------------------------------------------------

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nFinal Ranking")

print(results_df)

# Save Final Results

results_df.to_csv(
    "Final_Model_Ranking.csv",
    index=False
)

print("\nExperiment Completed Successfully.")



These outputs correspond directly to the figures and tables presented in Chapter 4 of this dissertation.
