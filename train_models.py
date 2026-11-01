from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# UCI Breast Cancer Wisconsin (Diagnostic): 569 records, 30 numeric features.
# This dataset is bundled with scikit-learn; cite UCI as the original public source.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target
target_names = {int(i): str(name) for i, name in enumerate(dataset.target_names)}

data = X.copy()
data["target"] = y
data.to_csv(ROOT / "full_dataset.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(probability=True, random_state=42))
    ]),
}

for display_name, model in models.items():
    model.fit(X_train, y_train)
    safe_name = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors": "knn",
        "Gaussian Naive Bayes": "naive_bayes",
        "Support Vector Machine": "svm",
    }[display_name]
    joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

# The assignment's app expects the held-out test CSV to include all features and target.
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv(ROOT / "test_data.csv", index=False)

metadata = {
    "feature_names": list(X.columns),
    "target_column": "target",
    "target_names": target_names,
    "dataset_name": "UCI Breast Cancer Wisconsin (Diagnostic)",
    "rows": int(data.shape[0]),
    "features": int(X.shape[1]),
    "test_rows": int(test_df.shape[0]),
    "random_state": 42,
    "test_size": 0.20
}
with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Training complete.")
print(f"Rows: {len(data)} | Features: {X.shape[1]} | Held-out test rows: {len(test_df)}")
print("Created: full_dataset.csv, test_data.csv, model/*.joblib, model/metadata.json")
print("Next: run streamlit run app.py")
