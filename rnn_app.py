import streamlit as st
import numpy as np
import pickle
import re
import string
import onnxruntime as ort
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    layout="wide"
)

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

session = ort.InferenceSession(
    "mental_health_rnn_model.onnx",
    providers=["CPUExecutionProvider"]
)

max_length = 50

stop_words = {
    "i","me","my","myself","we","our","ours","you","your",
    "yours","he","him","his","she","her","it","they",
    "them","their","what","which","who","this","that",
    "am","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","a","an","the",
    "and","but","if","or","because","as","until","while",
    "of","at","by","for","with","about","against","between",
    "into","through","during","before","after","above","below",
    "to","from","up","down","in","out","on","off","over",
    "under"
}

def clean_text(text):

    text = str(text).lower()

    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    tokens = text.split()

    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

def predict_sentiment(text):

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    padded = padded.astype(np.int32)

    input_name = session.get_inputs()[0].name

    prediction = session.run(
        None,
        {input_name: padded}
    )[0]

    predicted_class = np.argmax(prediction)

    confidence = prediction[0][predicted_class]

    sentiment = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    return sentiment, confidence, prediction[0]

st.title("AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

st.markdown("---")

st.header("About the Project")

st.write("""
This project uses Artificial Intelligence and Natural Language Processing
to analyze emotional sentiment from user text messages.

Emotional AI helps in understanding mental wellness patterns,
identifying emotional distress, and supporting early intervention.

Natural Language Processing (NLP) enables computers to understand
human language.

Simple Recurrent Neural Networks (RNNs) are used for sequence learning,
making them effective for sentiment analysis tasks.
""")

st.markdown("---")

st.header("Enter Your Thoughts")

user_input = st.text_area(
    "User Text Input",
    placeholder="Enter your thoughts or feelings here...",
    height=180
)

st.write("### Sample Sentences")

st.write("- I feel anxious and nervous about my future")
st.write("- Today was amazing and I feel very joyful")
st.write("- I am feeling lonely and emotionally tired")
st.write("- I feel calm, confident, and positive today")

if st.button("Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text")

    else:

        sentiment, confidence, probabilities = predict_sentiment(
            user_input
        )

        st.markdown("---")

        st.header("Prediction Output")

        st.success(f"Emotion Detected: {sentiment}")

        st.info(f"Confidence Score: {confidence * 100:.2f}%")

        if confidence > 0.80:
            status = "Strong Emotional Signal"
        elif confidence > 0.50:
            status = "Moderate Emotional Signal"
        else:
            status = "Weak Emotional Signal"

        st.write(f"Emotional Status: {status}")

        st.markdown("---")

        st.header("Sentiment Confidence Visualization")

        labels = label_encoder.classes_

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(labels, probabilities)

        ax.set_xlabel("Emotion")

        ax.set_ylabel("Probability")

        ax.set_title("Emotion Confidence Scores")

        plt.xticks(rotation=20)

        st.pyplot(fig)

        st.markdown("---")

        st.header("Emotional Wellness Guidance")

        if sentiment.lower() in ["depression", "sadness", "anxiety"]:

            st.error("""
Take a short break and talk with someone you trust.

Practice deep breathing, hydration, and light physical activity.

Remember that emotional challenges are temporary,
and seeking support is a sign of strength.
""")

        elif sentiment.lower() in ["normal", "happy", "positive"]:

            st.success("""
Great to see positive emotional signals.

Continue maintaining healthy routines,
social interaction, and positive thinking habits.
""")

        else:

            st.info("""
Stay mindful of your emotional well-being.

Regular sleep, exercise, and communication
can help improve mental wellness.
""")