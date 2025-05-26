import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from parser.preprocess import clean_text

# Load and clean data
df = pd.read_csv("data/resume.csv")
df['cleaned_resume'] = df['Resume_str'].apply(clean_text)

# Feature extraction
X = df['cleaned_resume']
y = df['Category']
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'models/resume_classifier.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
