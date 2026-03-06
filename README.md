# 📱 SMS Spam Detection

> A Machine Learning web application that classifies SMS messages as **Spam** or **Ham** using NLP techniques, TF-IDF vectorization, and an interactive Streamlit dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spam-sms-detecter.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Goal

The goal of this project is to automatically detect whether an SMS message is **spam** or **legitimate (ham)**. Rather than just training a model, the aim was to build a fully interactive dashboard where anyone can explore the data, compare model performance, and test live predictions in real time.

---

## 🚀 Live Demo

🔗 **[https://spam-sms-detecter.streamlit.app](https://spam-sms-detecter.streamlit.app)**

---

## ✨ App Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, dataset KPIs, and model leaderboard |
| 📊 **Dataset Explorer** | Filter, search, and browse all 5,573 messages with feature stats |
| 🔍 **EDA & Visualizations** | Word clouds, class distribution, top words, message length analysis |
| 🤖 **Model Comparison** | Bar charts, metric profiles, and confusion matrices for all 3 models |
| 🎯 **Predict a Message** | Live prediction with confidence score, probability chart and keyword highlights |
| 📂 **Upload & Predict CSV** | Batch predict on your own data and download results |

---

## 🧠 How It Works

1. Raw SMS messages are loaded from the dataset
2. Text is converted into numerical features using **TF-IDF Vectorization** (1,000 features)
3. Six hand-crafted features are extracted — exclamation marks, currency symbols, digits, uppercase count, character count, and word count
4. Three ML classifiers are trained and evaluated on the combined feature set
5. The best model predicts whether a message is **SPAM** or **HAM** along with a confidence score

---

## 📊 Model Results

| Rank | Model | Accuracy | F1 Score |
|------|-------|----------|----------|
| 🥇 | Naive Bayes | ~98% | ~93% |
| 🥈 | Logistic Regression | ~97% | ~91% |
| 🥉 | KNN (k=5, cosine) | ~95% | ~88% |

---

## 🗂️ Dataset

- **Source:** [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total Messages:** 5,573
- **Ham:** 4,825 (86.6%) · **Spam:** 747 (13.4%)

---

## 📁 Project Structure

```
sms-spam-detector/
├── app.py                  ← Main Streamlit application
├── spam.csv                ← Dataset
├── requirements.txt        ← Python dependencies
└── README.md               ← Project documentation
```

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/ishfaque2000/sms-spam-detector.git
cd sms-spam-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 👤 Author

**Ishfaque**
- GitHub: [@ishfaque2000](https://github.com/ishfaque2000)
- App: [spam-sms-detecter.streamlit.app](https://spam-sms-detecter.streamlit.app)
