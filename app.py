import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 style='text-align:center; color:#FFFFFF;'>📊 ML Model Comparison Dashboard</h1>", unsafe_allow_html=True)

# -------------------------------
# File Uploads
# -------------------------------
uploaded_file1 = st.sidebar.file_uploader("Upload Dataset 1", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Upload Dataset 2", type=["csv"])

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_data(df):
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    if 'Bare_Nuclei' in df.columns:
        df['Bare_Nuclei'] = df['Bare_Nuclei'].astype(int)

    return df

# -------------------------------
# Function to Train Model
# -------------------------------
def run_model(model, name, X_train, X_test, y_train, y_test, dataset_label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.markdown(f"<h3 style='color:#2196F3;'>{name} Results ({dataset_label})</h3>", unsafe_allow_html=True)
    st.write("Accuracy:", round(acc, 3))
    st.write("Precision:", round(prec, 3))
    st.write("Recall:", round(rec, 3))
    st.write("F1 Score:", round(f1, 3))

    with st.expander(f"Show Detailed Classification Report ({dataset_label})"):
        st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", ax=ax)
    ax.set_title(f"{name} Confusion Matrix ({dataset_label})")
    st.pyplot(fig)

    return acc

# -------------------------------
# Workflow if both datasets uploaded
# -------------------------------
if uploaded_file1 and uploaded_file2:
    df1 = preprocess_data(pd.read_csv(uploaded_file1))
    df2 = preprocess_data(pd.read_csv(uploaded_file2))

    st.subheader("Dataset 1 Preview")
    st.write(df1.head())
    st.subheader("Dataset 2 Preview")
    st.write(df2.head())

    # -------------------------------
    # Dataset Insights (EDA)
    # -------------------------------
    st.subheader("Dataset 1 Insights")
    st.write("Class Distribution:")
    st.bar_chart(df1.iloc[:, -1].value_counts())  # assumes target is last column
    st.write("Summary Statistics:")
    st.write(df1.describe())
    st.write("Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df1.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Dataset 2 Insights")
    st.write("Class Distribution:")
    st.bar_chart(df2.iloc[:, -1].value_counts())
    st.write("Summary Statistics:")
    st.write(df2.describe())
    st.write("Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df2.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    excluded = ["id", "Sample_code_number"]
    valid_targets1 = [col for col in df1.columns if col not in excluded]
    valid_targets2 = [col for col in df2.columns if col not in excluded]

    target1 = st.sidebar.selectbox("Select Target Column (Dataset 1)", valid_targets1)
    target2 = st.sidebar.selectbox("Select Target Column (Dataset 2)", valid_targets2)

    X1, y1 = df1.drop(columns=[target1]), df1[target1].astype(int)
    X2, y2 = df2.drop(columns=[target2]), df2[target2].astype(int)

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    scaler1, scaler2 = StandardScaler(), StandardScaler()
    X1_train, X1_test = scaler1.fit_transform(X1_train), scaler1.transform(X1_test)
    X2_train, X2_test = scaler2.fit_transform(X2_train), scaler2.transform(X2_test)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Logistic", "KNN", "Decision Tree", "Random Forest", "Comparison (Dataset 1)", "Cross-Dataset Comparison"
    ])

    with tab1:
        if st.button("Run Logistic Regression (Dataset 1)"):
            run_model(LogisticRegression(max_iter=1000), "Logistic Regression", X1_train, X1_test, y1_train, y1_test, "Dataset 1")
        if st.button("Run Logistic Regression (Dataset 2)"):
            run_model(LogisticRegression(max_iter=1000), "Logistic Regression", X2_train, X2_test, y2_train, y2_test, "Dataset 2")

    with tab2:
        if st.button("Run KNN (Dataset 1)"):
            run_model(KNeighborsClassifier(n_neighbors=5), "KNN", X1_train, X1_test, y1_train, y1_test, "Dataset 1")
        if st.button("Run KNN (Dataset 2)"):
            run_model(KNeighborsClassifier(n_neighbors=5), "KNN", X2_train, X2_test, y2_train, y2_test, "Dataset 2")

    with tab3:
        if st.button("Run Decision Tree (Dataset 1)"):
            run_model(DecisionTreeClassifier(), "Decision Tree", X1_train, X1_test, y1_train, y1_test, "Dataset 1")
        if st.button("Run Decision Tree (Dataset 2)"):
            run_model(DecisionTreeClassifier(), "Decision Tree", X2_train, X2_test, y2_train, y2_test, "Dataset 2")

    with tab4:
        if st.button("Run Random Forest (Dataset 1)"):
            run_model(RandomForestClassifier(n_estimators=100), "Random Forest", X1_train, X1_test, y1_train, y1_test, "Dataset 1")
        if st.button("Run Random Forest (Dataset 2)"):
            run_model(RandomForestClassifier(n_estimators=100), "Random Forest", X2_train, X2_test, y2_train, y2_test, "Dataset 2")

    with tab5:
        if st.button("Compare All Models (Dataset 1)"):
            models = {
                "Logistic": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100)
            }
            results = []
            for name, model in models.items():
                model.fit(X1_train, y1_train)
                y_pred = model.predict(X1_test)
                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y1_test, y_pred),
                    "Precision": precision_score(y1_test, y_pred, average="weighted"),
                    "Recall": recall_score(y1_test, y_pred, average="weighted"),
                    "F1 Score": f1_score(y1_test, y_pred, average="weighted")
                })
            results_df = pd.DataFrame(results)
            st.subheader("Comparison Table (Dataset 1)")
            st.write(results_df)
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x="Accuracy", y="Model", data=results_df, palette="coolwarm", ax=ax)
            ax.set_title("Model Comparison (Dataset 1)", fontsize=14, color="navy")
            st.pyplot(fig)

    with tab6:
        if st.button("Cross-Dataset Comparison"):
            models = {
                "Logistic": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100)
            }
            results = []
            for name, model in models.items():
                acc1 = run_model(model, name, X1_train, X1_test, y1_train, y1_test, "Dataset 1")
                acc2 = run_model(model, name, X2_train, X2_test)