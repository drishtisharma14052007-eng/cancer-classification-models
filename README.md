# 🧬 Cancer Classification Models

This project compares multiple machine learning models (Logistic Regression, KNN, Decision Tree, Random Forest) for breast cancer classification.  
It was built with **Python, scikit-learn, and Streamlit** to provide an interactive dashboard for dataset upload, preprocessing, model training, and evaluation.

---

## 🚀 Live Demo
👉 Try the app here: [Streamlit App]([https://your-username-your-repo.streamlit.app](https://cancer-classification-models-daefyjonguih5v2pf778bl.streamlit.app/#ml-model-comparison-dashboard))

---

## 📂 Project Structure
- `app.py` → Streamlit application
- `requirements.txt` → Dependencies for deployment
- `README.md` → Project documentation

---

## 📊 Features
- Upload CSV dataset
- Preprocessing (missing values, type conversion)
- Train/test split
- Model training and evaluation
- Confusion matrix visualization
- Accuracy comparison across models

---

## ⚙️ Workflow
1. **Upload Dataset**  
   Provide a CSV file with cancer data (e.g., UCI Breast Cancer dataset).

2. **Preprocessing**  
   - Replace missing values  
   - Convert categorical features to numeric  
   - Scale features for consistency  

3. **Model Training**  
   - Logistic Regression  
   - KNN  
   - Decision Tree  
   - Random Forest  

4. **Evaluation**  
   - Accuracy score  
   - Classification report  
   - Confusion matrix heatmap  

5. **Comparison**  
   - Side-by-side accuracy comparison  
   - Bar chart visualization  

---

## 📈 Results
The app displays:
- Accuracy for each model
- Confusion matrix heatmap
- Classification report (precision, recall, F1-score)
- Comparison chart of all models

---

## 🛠️ Installation (Local Run)
Clone the repo and install dependencies:
```bash
git clone https://github.com/your-username/cancer-classification-models.git
cd cancer-classification-models
pip install -r requirements.txt
streamlit run app.py
