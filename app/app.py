import streamlit as st
import joblib
import sys
import os
import fitz  # PyMuPDF for PDF text extraction
import numpy as np

# Added root path to import from parser/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser.preprocess import clean_text

# Initialize model and vectorizer variables and flags
model = None
vectorizer = None
model_loaded = False
vectorizer_loaded = False

# Load model with error handling
try:
    model = joblib.load('models/resume_classifier.pkl')
    model_loaded = True
except FileNotFoundError:
    st.error("Critical error: Failed to load the classification model (models/resume_classifier.pkl not found). Resume classification will not be available.")
except Exception as e:
    st.error(f"Critical error: An unexpected error occurred while loading the classification model: {e}. Resume classification will not be available.")

# Load vectorizer with error handling
try:
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    vectorizer_loaded = True
except FileNotFoundError:
    st.error("Critical error: Failed to load the TF-IDF vectorizer (models/tfidf_vectorizer.pkl not found). Resume classification will not be available.")
except Exception as e:
    st.error(f"Critical error: An unexpected error occurred while loading the TF-IDF vectorizer: {e}. Resume classification will not be available.")

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
    try:
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf_doc:
            resume_text += page.get_text()
    except Exception as e: # Using general Exception as fitz.INTERNALERROR might not be available
        st.error(f"Error processing the PDF file. It might be corrupted or in an unsupported format. Please try another file or paste the text manually. Details: {e}")
        resume_text = "" # Ensure resume_text is empty on error
elif not uploaded_file:
    resume_text = st.text_area("Or paste your resume content below:")

# Optional job description
job_description = st.text_area("Optional: Paste Job Description to compare")

# Determine if the button should be disabled
button_disabled = not (model_loaded and vectorizer_loaded)

if st.button("🔍 Classify Resume", disabled=button_disabled) and resume_text:
    if model_loaded and vectorizer_loaded:
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
        st.error("Classification cannot proceed as essential components (model/vectorizer) failed to load.")
elif button_disabled:
    st.warning("Resume classification is currently unavailable due to errors in loading necessary model or vectorizer files. Please check the error messages above.")
else:
    st.info("Please upload a PDF or enter resume text to begin.")