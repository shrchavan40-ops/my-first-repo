import pickle

# Step 1: Load the saved model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Step 2: Define a function to make predictions
def predict_sentiment(text):
    # Transform the input text
    text_vector = vectorizer.transform([text])
    
    # Get model prediction
    prediction = model.predict(text_vector)[0]
    
    # Interpret result
    if prediction == 1:
        return "Positive Sentiment 😊"
    else:
        return "Negative Sentiment 😞"

# Step 3: Test the function
if __name__ == "__main__":
    sample_text = input("Enter a sentence: ")
    result = predict_sentiment(sample_text)
    print("Predicted Sentiment:", result)
