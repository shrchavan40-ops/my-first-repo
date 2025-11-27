import pickle
import gradio as gr

# Load saved model & vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Prediction function (expected input: raw text)
def predict_sentiment(text):
    if not text or text.strip() == "":
        return "Please enter some text."
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "😊 Positive" if pred == 1 else "☹️ Negative"

# Week 4: Improved input components for text model
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Type a sentence or paragraph here...", label="Enter text"),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analysis App",
    description="Enter any text to predict sentiment (Positive / Negative).",
    examples=[
        ["I love this product! It works perfectly."],
        ["This is the worst experience I've had."],
        ["Not bad, could be improved."]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()

