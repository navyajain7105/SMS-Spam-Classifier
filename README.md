# Email & SMS Spam Classifier
This project is a machine learning model built to classify messages as "Spam" or "Ham" (not spam). It uses a soft-voting ensemble of three classifiers trained on TF-IDF vectorized text data.<br><br>

The project includes two main parts:

-->Email Spam Classifier.ipynb: A Jupyter Notebook detailing the complete end-to-end workflow, from data cleaning and exploratory data analysis (EDA) to model training and evaluation.

-->app.py: A lightweight, interactive web application built with Streamlit that deploys the trained model for real-time predictions.
<br><br>

## ✨ Features<br>
&nbsp;⦿ High Accuracy: Achieves 98% accuracy and 1.0 precision on the test set.

&nbsp;⦿ Ensemble Model: Uses a VotingClassifier (SVC, Multinomial Naive Bayes, Extra Trees) for robust and reliable predictions.

&nbsp;⦿ Interactive UI: The Streamlit app provides a simple interface to enter any message and get an instant classification.

&nbsp;⦿ NLP Pipeline: Implements a comprehensive text preprocessing pipeline using NLTK for cleaning and feature engineering.

&nbsp;⦿ Data-Driven: Includes detailed EDA with visualizations like word clouds and correlation heatmaps to understand the dataset.

## 📁 Repository Structure
.<br>
├── app.py                &nbsp;&nbsp;  # The Streamlit web application<br>
├── Email Spam Classifier.ipynb  &nbsp;&nbsp;# Jupyter Notebook for analysis and model training<br>
├── model.pkl               &nbsp;&nbsp;# Pickled file for the trained VotingClassifier<br>
├── vectorizer.pkl         &nbsp;&nbsp; # Pickled file for the TfidfVectorizer<br>
├── mail_data.csv         &nbsp;&nbsp;  # The raw dataset used for training<br>
├── requirements.txt      &nbsp;&nbsp;  # Python dependencies<br>
└── README.md             &nbsp;&nbsp;  # You are here!<br>

## 🛠️ Workflow & Methodology
1. Data Cleaning & EDA

2. Text Preprocessing

3. Model Building

4. Web Application

## 🔧 How to Run Locally
1. Clone the Repository
```
git clone https://github.com/navyajain7105/SMS-Spam-Classifier.git
```
2. Create a Virtual Environment (Recommended)
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Run the Streamlit App
```
streamlit run app.py
```

