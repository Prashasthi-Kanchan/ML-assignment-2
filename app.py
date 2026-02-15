import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# ----------------------------------
# Load Saved Models
# ----------------------------------

models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

scaler = joblib.load("model/scaler.pkl")

# ----------------------------------
# Streamlit UI
# ----------------------------------

st.title("Adult Income Classification App")
st.write("Predict whether income is >50K or <=50K")

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

# Model selection
model_choice = st.selectbox(
    "Select Model",
    list(models.keys())
)

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview:")
    st.dataframe(df.head())

    # Preprocessing
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    if "income" in df.columns:
        df["income"] = df["income"].map({
            "<=50K": 0,
            ">50K": 1,
            "<=50K.": 0,
            ">50K.": 1
        })

        X = df.drop("income", axis=1)
        y = df["income"]

        X = pd.get_dummies(X, drop_first=True)

        # Align columns with training (important)
        model = models[model_choice]

        try:
            X_scaled = scaler.transform(X)
        except:
            st.error("Feature mismatch! Please upload correct test file.")
            st.stop()

        # Predictions
        y_pred = model.predict(X_scaled)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:,1]
            auc = roc_auc_score(y, y_prob)
        else:
            auc = "N/A"

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", round(accuracy, 4))
        st.write("Precision:", round(precision, 4))
        st.write("Recall:", round(recall, 4))
        st.write("F1 Score:", round(f1, 4))
        st.write("AUC:", auc)
        st.write("MCC:", round(mcc, 4))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    else:
        st.error("Uploaded file must contain 'income' column.")
