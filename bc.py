import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Title
# -------------------------------
st.title("Machine Learning Model Comparison System")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # -------------------------------
    # Preprocessing Function
    # -------------------------------
    def preprocess_data(df):
        df.replace('?', pd.NA, inplace=True)
        df.dropna(inplace=True)

        if 'Bare_Nuclei' in df.columns:
            df['Bare_Nuclei'] = df['Bare_Nuclei'].astype(int)

        return df

    df = preprocess_data(df)

    # -------------------------------
    # Feature Selection
    # -------------------------------
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("Run Algorithms")

    col1, col2, col3, col4 = st.columns(4)

    # -------------------------------
    # Function to Train Model
    # -------------------------------
    def run_model(model, name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.subheader(f"{name} Results")
        st.write("Accuracy:", round(acc, 3))
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        st.pyplot(fig)

        return acc

    # -------------------------------
    # Buttons
    # -------------------------------
    if col1.button("Logistic"):
        run_model(LogisticRegression(max_iter=1000), "Logistic Regression")

    if col2.button("KNN"):
        run_model(KNeighborsClassifier(n_neighbors=5), "KNN")

    if col3.button("Decision Tree"):
        run_model(DecisionTreeClassifier(), "Decision Tree")

    if col4.button("Random Forest"):
        run_model(RandomForestClassifier(n_estimators=100), "Random Forest")

    # -------------------------------
    # Compare All Models
    # -------------------------------
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

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred)
            })

        results_df = pd.DataFrame(results)

        st.subheader("Comparison Table")
        st.write(results_df)

        # Plot
        fig, ax = plt.subplots()
        ax.bar(results_df["Model"], results_df["Accuracy"])
        ax.set_title("Model Comparison")
        st.pyplot(fig)