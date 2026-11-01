from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Classification Model Evaluation", page_icon="📊", layout="wide")
st.title("Machine Learning Classification Model Comparison")
st.write(
    "Upload a **held-out test CSV** containing all required input features and the `target` column. "
    "This app loads pre-trained models; it does not retrain them on uploaded data."
)

with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_names"]
TARGET = metadata["target_column"]
TARGET_NAMES = {int(k): v for k, v in metadata["target_names"].items()}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.sidebar.header("Evaluation settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

default_path = ROOT / "test_data.csv"
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif default_path.exists():
    test_df = pd.read_csv(default_path)
    st.info("Showing the included held-out test_data.csv. Upload another CSV to evaluate it instead.")
else:
    st.warning("Upload a CSV to begin.")
    st.stop()

missing = [col for col in FEATURES + [TARGET] if col not in test_df.columns]
if missing:
    st.error("The CSV is missing required columns: " + ", ".join(missing))
    st.stop()
if test_df.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

try:
    X_test = test_df[FEATURES].apply(pd.to_numeric, errors="raise")
    y_test = pd.to_numeric(test_df[TARGET], errors="raise").astype(int)
except Exception as exc:
    st.error(f"Features and target must be numeric for this dataset. Details: {exc}")
    st.stop()

if X_test.isnull().any().any() or y_test.isnull().any():
    st.error("The CSV contains missing values in the features or target.")
    st.stop()

model_path = MODEL_DIR / MODEL_FILES[model_name]
try:
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
except Exception as exc:
    st.error(f"Model prediction failed. Check that the CSV uses the same feature format as training. Details: {exc}")
    st.stop()

classes = sorted(pd.Series(y_test).unique().tolist())
if len(classes) < 2:
    st.error("AUC cannot be calculated because the uploaded target contains only one class. Include examples from both classes.")
    st.stop()

try:
    # Binary classification: score for positive class 1.
    if len(classes) == 2:
        model_classes = list(model.classes_)
        positive_label = 1 if 1 in model_classes else model_classes[-1]
        positive_index = model_classes.index(positive_label)
        auc = roc_auc_score(y_test, y_score[:, positive_index])
        precision = precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    else:
        auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
except Exception as exc:
    st.error(f"Could not calculate AUC. Check target labels and class coverage. Details: {exc}")
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC Score": auc,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "MCC Score": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Results: {model_name}")
st.caption(f"Evaluated {len(test_df)} rows · {len(FEATURES)} input features")
cols = st.columns(3)
for i, (label, value) in enumerate(metrics.items()):
    cols[i % 3].metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    labels = sorted(set(y_test.tolist()) | set(pd.Series(y_pred).tolist()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(labels)), [TARGET_NAMES.get(int(x), str(x)) for x in labels], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), [TARGET_NAMES.get(int(x), str(x)) for x in labels])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_test, y_pred,
        labels=sorted(set(y_test.tolist()) | set(pd.Series(y_pred).tolist())),
        target_names=[TARGET_NAMES.get(int(x), str(x)) for x in sorted(set(y_test.tolist()) | set(pd.Series(y_pred).tolist()))],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")

with st.expander("Preview uploaded test data"):
    st.dataframe(test_df.head(20), width="stretch")

st.subheader("Compare all five models")
if st.button("Evaluate all five models on this test CSV"):
    rows = []
    for name, filename in MODEL_FILES.items():
        fitted = joblib.load(MODEL_DIR / filename)
        pred = fitted.predict(X_test)
        prob = fitted.predict_proba(X_test)
        if len(classes) == 2:
            model_classes = list(fitted.classes_)
            positive_label = 1 if 1 in model_classes else model_classes[-1]
            positive_index = model_classes.index(positive_label)
            model_auc = roc_auc_score(y_test, prob[:, positive_index])
            p = precision_score(y_test, pred, pos_label=positive_label, zero_division=0)
            r = recall_score(y_test, pred, pos_label=positive_label, zero_division=0)
            f = f1_score(y_test, pred, pos_label=positive_label, zero_division=0)
        else:
            model_auc = roc_auc_score(y_test, prob, multi_class="ovr", average="weighted")
            p = precision_score(y_test, pred, average="weighted", zero_division=0)
            r = recall_score(y_test, pred, average="weighted", zero_division=0)
            f = f1_score(y_test, pred, average="weighted", zero_division=0)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": model_auc,
            "Precision": p,
            "Recall": r,
            "F1": f,
            "MCC": matthews_corrcoef(y_test, pred)
        })
    comparison = pd.DataFrame(rows).set_index("Model").round(4)
    st.dataframe(comparison, width="stretch")
    st.download_button(
        "Download comparison table (CSV)",
        comparison.to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv"
    )

st.caption("Important: evaluate with labelled test data that was not used to train the saved models.")
