import streamlit as st
import numpy as np
import pickle
import re
import string
import onnxruntime as ort
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    layout="wide"
)

with open("word_index.pkl", "rb") as file:
    word_index = pickle.load(file)

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

    return tokens

def text_to_sequence(tokens):

    sequence = []

    for word in tokens:

        if word in word_index:

            sequence.append(word_index[word])

        else:

            sequence.append(1)

    return sequence

def custom_pad_sequences(sequence, maxlen):

    padded = np.zeros((1, maxlen), dtype=np.int64)

    length = min(len(sequence), maxlen)

    padded[0, :length] = sequence[:length]

    return padded

def predict_sentiment(text):

    cleaned_tokens = clean_text(text)

    sequence = text_to_sequence(cleaned_tokens)

    padded = custom_pad_sequences(
        sequence,
        max_length
    )

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

Emotional AI helps monitor mental wellness,
identify emotional distress,
and support early intervention.

Natural Language Processing (NLP)
allows computers to understand human language.

Simple Recurrent Neural Networks (RNNs)
learn sequential emotional patterns from text data.
""")

st.markdown("---")

st.header("User Text Input Area")

user_input = st.text_area(
    "Enter your thoughts",
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

        st.header("Visualization Area")

        labels = label_encoder.classes_

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(labels, probabilities)

        ax.set_xlabel("Emotion")

        ax.set_ylabel("Probability")

        ax.set_title("Emotion Confidence Graph")

        plt.xticks(rotation=20)

        st.pyplot(fig)

        st.markdown("---")

        st.header("Emotional Guidance Area")

        if sentiment.lower() in ["depression", "anxiety", "sadness"]:

            st.error("""
Take a short break and talk with someone you trust.

Practice breathing exercises,
stay hydrated,
and try light physical activity.

Remember:
Seeking support is a sign of strength.
""")

        elif sentiment.lower() in ["normal", "happy", "positive"]:

            st.success("""
Great to see positive emotional signals.

Maintain healthy routines,
exercise regularly,
and continue positive social interaction.
""")

        else:

            st.info("""
Stay mindful of your emotional wellness.

Good sleep, communication,
and relaxation can improve emotional balance.
""")
