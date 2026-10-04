# ============================================================
# WEEK 6 - TELECOM CUSTOMER CHURN CAPSTONE PROJECT
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 1. CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("Telco-Customer-Churn.csv")

print("=" * 60)
print("TELECOM CUSTOMER CHURN CAPSTONE PROJECT")
print("=" * 60)

print("\nOriginal Dataset Shape:", df.shape)

# ------------------------------------------------------------
# 3. DATA CLEANING
# ------------------------------------------------------------

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values after conversion
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Fill missing TotalCharges using median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove customer ID because it is not useful for prediction
df.drop("customerID", axis=1, inplace=True)

# Convert SeniorCitizen to categorical
df["SeniorCitizen"] = df["SeniorCitizen"].map({
    0: "No",
    1: "Yes"
})

# Save cleaned dataset
df.to_csv(
    "outputs/cleaned_telco_customer_churn.csv",
    index=False
)

print("\nCleaned Dataset Shape:", df.shape)
print("Cleaning completed successfully.")

# ------------------------------------------------------------
# 4. BASIC STATISTICS
# ------------------------------------------------------------

print("\nDataset Statistics:")
print(df.describe())

# ------------------------------------------------------------
# 5. EDA - CHURN DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/graphs/churn_distribution.png")
plt.close()

# ------------------------------------------------------------
# 6. EDA - CONTRACT VS CHURN
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)
plt.title("Churn Distribution by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/graphs/contract_vs_churn.png")
plt.close()

# ------------------------------------------------------------
# 7. EDA - TENURE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True
)
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/graphs/tenure_vs_churn.png")
plt.close()

# ------------------------------------------------------------
# 8. EDA - MONTHLY CHARGES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("outputs/graphs/monthly_charges_vs_churn.png")
plt.close()

# ------------------------------------------------------------
# 9. EDA - INTERNET SERVICE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)
plt.title("Internet Service vs Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/graphs/internet_service_vs_churn.png")
plt.close()

# ------------------------------------------------------------
# 10. EDA - PAYMENT METHOD
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)
plt.title("Payment Method vs Churn")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("outputs/graphs/payment_method_vs_churn.png")
plt.close()

# ------------------------------------------------------------
# 11. CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/graphs/correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------
# 12. PREPARE DATA FOR MACHINE LEARNING
# ------------------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

categorical_features = X.select_dtypes(
    include=["object"]
).columns

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# ------------------------------------------------------------
# 13. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 14. LOGISTIC REGRESSION
# ------------------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 15. RANDOM FOREST
# ------------------------------------------------------------

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)
rf_prob = random_forest.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 16. MODEL EVALUATION
# ------------------------------------------------------------

results = []

models = {
    "Logistic Regression": (
        logistic_pred,
        logistic_prob
    ),
    "Random Forest": (
        rf_pred,
        rf_prob
    )
}

for model_name, (pred, prob) in models.items():

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, prob)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            pred
        )
    )

# Save model comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# ------------------------------------------------------------
# 17. MODEL COMPARISON GRAPH
# ------------------------------------------------------------

results_melted = results_df.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=results_melted,
    x="Metric",
    y="Score",
    hue="Model"
)
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(
    "outputs/graphs/model_comparison.png"
)
plt.close()

# ------------------------------------------------------------
# 18. CONFUSION MATRIX - BEST MODEL
# ------------------------------------------------------------

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

if best_model_name == "Logistic Regression":
    best_pred = logistic_pred
else:
    best_pred = rf_pred

cm = confusion_matrix(
    y_test,
    best_pred
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title(
    f"Confusion Matrix - {best_model_name}"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(
    "outputs/graphs/confusion_matrix.png"
)
plt.close()

# ------------------------------------------------------------
# 19. K-MEANS CUSTOMER SEGMENTATION
# ------------------------------------------------------------

cluster_data = df[
    [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
].copy()

scaler = StandardScaler()

cluster_scaled = scaler.fit_transform(
    cluster_data
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    cluster_scaled
)

df["Cluster"] = clusters

# Save clustered data
df.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

# ------------------------------------------------------------
# 20. PCA VISUALIZATION OF CLUSTERS
# ------------------------------------------------------------

pca = PCA(
    n_components=2,
    random_state=42
)

pca_data = pca.fit_transform(
    cluster_scaled
)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=clusters,
    palette="viridis",
    s=50
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/customer_clusters.png"
)

plt.close()

# ------------------------------------------------------------
# 21. CLUSTER SUMMARY
# ------------------------------------------------------------

cluster_summary = df.groupby(
    "Cluster"
)[
    [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
].mean()

cluster_summary["CustomerCount"] = (
    df["Cluster"].value_counts()
)

cluster_summary.to_csv(
    "outputs/cluster_summary.csv"
)

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION")
print("=" * 60)

print(cluster_summary)

# ------------------------------------------------------------
# 22. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nBest Model:", best_model_name)

print("\nGenerated Files:")
print("1. cleaned_telco_customer_churn.csv")
print("2. model_comparison.csv")
print("3. customer_segments.csv")
print("4. cluster_summary.csv")

print("\nGenerated Graphs:")
print("1. churn_distribution.png")
print("2. contract_vs_churn.png")
print("3. tenure_vs_churn.png")
print("4. monthly_charges_vs_churn.png")
print("5. internet_service_vs_churn.png")
print("6. payment_method_vs_churn.png")
print("7. correlation_heatmap.png")
print("8. model_comparison.png")
print("9. confusion_matrix.png")
print("10. customer_clusters.png")

print("\nAll Week 6 capstone tasks completed.")