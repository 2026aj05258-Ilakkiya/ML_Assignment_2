# ML Assignment 2 — Classification Models

## a. Problem Statement

The objective of this project is to implement and compare five supervised machine learning classification algorithms using a public classification dataset. The models are evaluated using six performance metrics to identify the most suitable classifier for the selected dataset.

The project also includes an interactive Streamlit web application that allows users to upload labelled test data, select a trained classification model and view its evaluation results. The application uses previously saved models without retraining them on the uploaded test data.

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)

**Dataset Source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Dataset Loading:** The dataset is obtained using `sklearn.datasets.load_breast_cancer`, which provides the Breast Cancer Wisconsin Diagnostic dataset through Scikit-learn.

**Dataset Characteristics:**

- Number of instances: 569
- Number of input features: 30
- Classification type: Binary classification
- Target variable: Diagnosis
- Target encoding: 0 = Malignant, 1 = Benign

The dataset contains numerical features computed from digitized images of breast mass samples. These features are used to classify tumours as malignant or benign.

The dataset is divided into training and testing sets using an 80:20 split with stratification to maintain the class distribution. The test dataset contains 114 records and is used to evaluate the trained models.

## c. GitHub Repository Link

**GitHub Repository:**  
https://github.com/2026aj05258-Ilakkiya/ML_Assignment_2

The repository contains the complete project source code, requirements file, README documentation, dataset files, saved trained models and preprocessing metadata required for the Streamlit application.

## d. Models Used and Evaluation

Five supervised machine learning classification algorithms are implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using the following six performance metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC Score:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures the proportion of correctly predicted positive instances among all predicted positive instances.
- **Recall:** Measures the proportion of actual positive instances correctly identified.
- **F1 Score:** Represents the harmonic mean of Precision and Recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications by considering all four confusion matrix categories.

### Model Performance Comparison

The following table presents the evaluation results obtained on the held-out test dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 98.25% and the highest AUC score of 0.9954. It also achieved Precision, Recall and F1 scores of 0.9861, along with an MCC score of 0.9623. These results demonstrate excellent classification performance on the test dataset.

**2. Decision Tree Classifier**

The Decision Tree classifier achieved an accuracy of 92.11% and an AUC score of 0.9163. Its performance was comparatively lower than the other models across most evaluation metrics. However, it achieved a Precision score of 0.9565.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 95.61% and an AUC score of 0.9788. Its Recall score of 0.9722 indicates good identification of positive-class instances. It also achieved an F1 score of 0.9655 and an MCC score of 0.9054.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 93.86% and an AUC score of 0.9878. Although its AUC score was high, its Accuracy and MCC scores were lower than those of Logistic Regression and SVM. It achieved an F1 score of 0.9517.

**5. Support Vector Machine (SVM)**

SVM achieved an accuracy of 98.25% and an AUC score of 0.9950. Its Precision, Recall, F1 and MCC scores were the same as those obtained by Logistic Regression. This indicates that SVM also performed exceptionally well on the selected dataset.

### Overall Winner

**Logistic Regression is the overall winner for the selected Breast Cancer Wisconsin Diagnostic dataset.**

It achieved the highest AUC score of 0.9954 and an accuracy of 98.25%. It also matched Support Vector Machine in Precision, Recall, F1 and MCC scores. Based on the overall evaluation, Logistic Regression demonstrated the strongest performance among the five implemented classification models.

## e. Streamlit Application

The project includes an interactive web application developed using Streamlit and deployed on Streamlit Community Cloud.

**Live Streamlit Application:**  
https://mlassignment2-2026aj05258.streamlit.app

### Application Features

1. **CSV Dataset Upload:** Allows users to upload a labelled test dataset in CSV format containing all required input features and the target column.
2. **Model Selection:** Provides a dropdown menu to select one of the five trained classification models.
3. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score and MCC for the selected model.
4. **Confusion Matrix and Classification Report:** Displays classification evaluation results for the uploaded test dataset.
5. **Model Comparison:** Allows users to evaluate and compare the performance of all five saved classification models.

The application loads the previously trained models from the `model/` directory. It evaluates the uploaded test dataset without retraining the models.

## f. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── full_dataset.csv
├── .gitignore
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metadata.json
```

### Description of Project Files

- **app.py:** Implements the Streamlit application, CSV upload functionality, model selection, evaluation metrics, confusion matrix, classification report and model comparison.
- **train_models.py:** Handles dataset preparation, train-test splitting, model training, evaluation and saving the trained models.
- **requirements.txt:** Contains the Python libraries and dependencies required to run the project.
- **README.md:** Provides project documentation, dataset information, model performance comparison and observations.
- **test_data.csv:** Contains the held-out test dataset used for evaluating the trained models.
- **full_dataset.csv:** Contains the complete dataset used in the project.
- **.gitignore:** Specifies files and directories that should be excluded from version control.
- **model/:** Stores all five saved trained classification models and the metadata required by the application.

## g. Reproducibility and Academic Integrity

The train-test split uses stratification and a fixed `random_state=42` to support reproducibility.

All five classification models are trained in `train_models.py` and saved for later use. The Streamlit application loads these saved models and evaluates the uploaded labelled test dataset without retraining.

The project implementation, model evaluation and application functionality have been reviewed and tested. The code and results are documented to support understanding and reproducibility.

## h. Conclusion

This project demonstrates the implementation, evaluation and comparison of five supervised machine learning classification algorithms using the Breast Cancer Wisconsin Diagnostic dataset.

Based on the evaluation results, Logistic Regression achieved the best overall performance, with an accuracy of 98.25% and an AUC score of 0.9954. Support Vector Machine also demonstrated excellent performance, while KNN and Gaussian Naive Bayes achieved good classification results. The Decision Tree classifier showed comparatively lower performance.

The Streamlit application provides an interactive interface for uploading test data, selecting trained models and viewing their evaluation results. This completes the machine learning workflow from model development and evaluation to application deployment.
