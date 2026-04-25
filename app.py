import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
# File Upload
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # -------------------------------
    # Preprocessing Function
    # -------------------------------
    def preprocess_data(df):
        df.replace('?', np.nan, inplace=True)
        df.dropna(inplace=True)

        if 'Bare_Nuclei' in df.columns:
            df['Bare_Nuclei'] = df['Bare_Nuclei'].astype(int)

        return df

    df = preprocess_data(df)

    # -------------------------------
    # Feature Selection
    # -------------------------------
    excluded = ["id", "Sample_code_number"]
    valid_targets = [col for col in df.columns if col not in excluded]

    target = st.sidebar.selectbox("Select Target Column", valid_targets)

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Scale Features
    # -------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # Function to Train Model
    # -------------------------------
    def run_model(model, name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.markdown(f"<h3 style='color:#2196F3;'>{name} Results</h3>", unsafe_allow_html=True)
        st.write("Accuracy:", round(acc, 3))

        with st.expander("Show Detailed Classification Report"):
            st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        st.pyplot(fig)

        return acc

    # -------------------------------
    # Tabs for Models
    # -------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Logistic", "KNN", "Decision Tree", "Random Forest", "Comparison"])

    with tab1:
        if st.button("Run Logistic Regression"):
            run_model(LogisticRegression(max_iter=1000), "Logistic Regression")

    with tab2:
        if st.button("Run KNN"):
            run_model(KNeighborsClassifier(n_neighbors=5), "KNN")

    with tab3:
        if st.button("Run Decision Tree"):
            run_model(DecisionTreeClassifier(), "Decision Tree")

    with tab4:
        if st.button("Run Random Forest"):
            run_model(RandomForestClassifier(n_estimators=100), "Random Forest")

    with tab5:
        if st.button("Compare All Models"):
            models = {
                "Logistic": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100)
            }

            results = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results.append({"Model": name, "Accuracy": accuracy_score(y_test, y_pred)})

            results_df = pd.DataFrame(results)

            st.subheader("Comparison Table")
            st.write(results_df)

            # Attractive Chart
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x="Accuracy", y="Model", data=results_df, palette="coolwarm", ax=ax)
            ax.set_title("Model Comparison", fontsize=14, color="navy")
            st.pyplot(fig)
