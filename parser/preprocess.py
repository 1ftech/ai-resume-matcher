import re

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)       # Remove punctuation
    text = re.sub(r'\d+', '', text)           # Remove numbers
    text = re.sub(r'\s+', ' ', text)          # Remove extra spaces
    return text.lower()
