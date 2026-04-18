import streamlit as st
import pickle

# Load
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🧠 AI Fake News Detector")
st.write("Paste a news article below to check if it's real or fake.")

text = st.text_area("Enter News Article")

if st.button("Check"):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    if pred == 1:
        st.success(f"REAL ✅ Confidence: {prob:.2f}")
    else:
        st.error(f"FAKE ❌ Confidence: {prob:.2f}")