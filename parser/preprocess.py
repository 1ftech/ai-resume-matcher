import re
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')

def clean_text(text):
    # Preserve letters, numbers, whitespace, and '.', '+', '#', '-'
    text = re.sub(r'[^a-zA-Z0-9\s.+#-]', '', text)
    text = re.sub(r'\d+', '', text)           # Remove numbers (optional, consider if numbers in like C++ are important)
    text = text.lower()                       # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)          # Remove extra spaces

    # Tokenize and remove stopwords
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)
