import streamlit as st
import pickle
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# --- Text Transformation Function (Logic remains the same) ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# --- Load Model and Vectorizer ---
# Use the correct file paths for deployment
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files (vectorizer.pkl or model.pkl) not found. Please ensure they are in the correct directory.")
    st.stop()


# =======================================================
#             STREAMLIT UI/UX IMPROVEMENTS
# =======================================================

# --- Page Configuration (Optional but improves look) ---
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Header Section ---
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .stTextArea label {
        font-size: 1.15rem;
        font-weight: 500;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.1rem;
        height: 3em;
        background-color: #4CAF50; /* Green */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📧 SMS Spam Classifier")
st.markdown("---")

# --- Input Area ---
input_sms = st.text_area(
    "Enter the message you want to classify:",
    height=200,
    placeholder="Type or paste your message here...",
)

# --- Prediction Button ---
if st.button('Classify Message'):

    if input_sms:
        with st.spinner('Analyzing message...'):
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            
            # 2. Vectorize (NOTE: Added .toarray() for SVC compatibility)
            vector_input = tfidf.transform([transformed_sms]).toarray()
            
            # 3. Predict
            result = model.predict(vector_input)[0]

            # 4. Display Results
            st.markdown("---")
            
            if result == 1:
                st.error("🔴 Prediction: SPAM")
                st.balloons()
            else:
                st.success("✅ Prediction: NOT SPAM (HAM)")
            
            st.info(f"Confidence (Spam=1, Ham=0): {model.predict_proba(vector_input)[0][result]:.2f}")
    else:
        st.warning("Please enter a message to proceed with classification.")

# --- Footer/Instructions ---
st.markdown("""
<br><br>
**How it works:**
This classifier uses a Voting Classifier (a combination of SVC, Multinomial Naive Bayes, and Extra Trees Classifier) trained on a vast SMS dataset. It cleans the text (removes stop words, punctuation, and performs stemming) and converts it into a numerical format (TF-IDF) before making a prediction.
""", unsafe_allow_html=True)
