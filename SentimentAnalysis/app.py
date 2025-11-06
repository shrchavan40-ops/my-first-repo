import gradio as gr
import pickle

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define prediction function
def predict_sentiment(text):
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return "😊 Positive" if prediction == 1 else "😡 Negative"

# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence here..."),
    outputs="text",
    title="Sentiment Analysis App",
    description="Enter a sentence to analyze its sentiment (positive or negative).",
    examples=[
        ["I love this!"],
        ["This is terrible."],
        ["Not bad, could be better."],
    ]
)

iface.launch()

