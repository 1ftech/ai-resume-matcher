import streamlit as st
import joblib
import sys
import os
import fitz  # PyMuPDF for PDF text extraction
import numpy as np

# Added root path to import from parser/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser.preprocess import clean_text

# Load model and vectorizer
model = joblib.load('models/resume_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

st.set_page_config(page_title="Resume Role Classifier", layout="centered")
st.title("📄 AI Resume Role Classifier")

st.markdown("""
Upload your **resume as PDF** or paste the text manually below.
Optionally, paste a **job description** to compare similarity.
""")

# Resume input
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
resume_text = ""

if uploaded_file:
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf_doc:
        resume_text += page.get_text()
elif not uploaded_file:
    resume_text = st.text_area("Or paste your resume content below:")

# Optional job description
job_description = st.text_area("Optional: Paste Job Description to compare")

if st.button("🔍 Classify Resume") and resume_text:
    clean = clean_text(resume_text)
    vec = vectorizer.transform([clean])

    # Predict top 3 roles
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    top_indices = np.argsort(proba)[-3:][::-1]
    st.subheader("🎯 Top 3 Predicted Roles")
    for idx in top_indices:
        st.write(f"- **{classes[idx]}** ({proba[idx]*100:.2f}% confidence)")

    # Optional: Compare to job description
    if job_description:
        from sklearn.metrics.pairwise import cosine_similarity
        job_clean = clean_text(job_description)
        job_vec = vectorizer.transform([job_clean])
        similarity = cosine_similarity(vec, job_vec)[0][0]
        st.subheader("📌 Resume vs Job Description Similarity")
        st.write(f"Cosine Similarity: **{similarity:.2f}** (0 = no match, 1 = perfect match)")
else:
    st.info("Please upload a PDF or enter resume text to begin.")