import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample data
data = {
    "text": [
        "I love this product!",
        "This is amazing",
        "I hate this item",
        "This is the worst purchase ever",
        "I am very happy",
        "I am not satisfied"
    ],
    "label": [1, 1, 0, 0, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer correctly (in binary mode)
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")

