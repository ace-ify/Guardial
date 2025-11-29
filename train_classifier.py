# train_classifier.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib # For saving/loading models
import os

# Load the dataset
# Adjust path if your data is elsewhere
try:
    df = pd.read_csv('data/train.csv')
except FileNotFoundError:
    print("Error: data/train.csv not found. Please download it from a dataset (e.g., Kaggle Toxic Comment Classification) and place it in data/train.csv")
    raise SystemExit(1)

# Expected toxicity columns
toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
missing = [c for c in toxic_cols if c not in df.columns]
if missing:
    print(f"Error: dataset is missing expected columns: {missing}")
    print("Make sure you're using a compatible toxic comments dataset with the columns:\n", toxic_cols)
    raise SystemExit(1)

# Simplify for binary classification: "toxic" vs. "not toxic"
# Combine all toxicity labels into one 'is_toxic' column
df['is_toxic'] = df[toxic_cols].any(axis=1).astype(int)

# Drop NaN comment_text rows
df = df.dropna(subset=['comment_text'])

# Use a small sample if dataset is very large (optional)
# df = df.sample(n=100000, random_state=42)  # uncomment to limit size

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['comment_text'], df['is_toxic'], test_size=0.2, random_state=42
)

# Create a pipeline: TF-IDF Vectorizer + Logistic Regression Classifier
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)), # Limit features for simplicity
    ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
])

# Train the model
print("Training harmful content classifier...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Evaluate the model (optional, but good practice)
y_pred = model_pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model_pipeline, os.path.join(model_dir, 'harmful_content_model.joblib'))
print(f"Model saved to {os.path.join(model_dir, 'harmful_content_model.joblib')}")
