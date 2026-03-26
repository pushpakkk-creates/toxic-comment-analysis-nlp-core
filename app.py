import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[\w]*', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)

# UI
st.set_page_config(page_title="Toxic Comment Detector", layout="centered")

st.title("💬 Toxic Comment Detection System")
st.markdown("### 🔍 Analyze text for toxicity using Machine Learning")

st.write("---")

# Input box
user_input = st.text_area("✏️ Enter your text below:")

# Predict button
if st.button("🚀 Analyze"):
    
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        # 🔥 Confidence (works for Logistic Regression)
        try:
            prob = model.predict_proba(vector)[0]
            confidence = max(prob)
        except:
            # fallback for models like SVM
            score = model.decision_function(vector)[0]
            confidence = abs(score)

        st.write("---")
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("⚠️ Toxic Comment Detected")
        else:
            st.success("✅ Non-Toxic Comment")

        # Probability display
        st.markdown(f"### 🔢 Confidence Score: `{confidence:.2f}`")

        # Progress bar
        st.progress(min(float(confidence), 1.0))

        # Extra info
        st.write("### 🧠 Processed Text:")
        st.code(cleaned)

        st.write("---")
        st.info("ℹ️ Note: Model uses TF-IDF, so some contextual errors may occur.")