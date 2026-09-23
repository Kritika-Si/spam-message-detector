import tensorflow as tf
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download("stopwords", quiet=True)

# Load trained model
model = tf.keras.models.load_model("spam_detection_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load English stopwords
stop_words = stopwords.words("english")


def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove stopwords and convert to lowercase
    words = []

    for word in text.split():
        word = word.lower()

        if word not in stop_words:
            words.append(word)

    return " ".join(words)


def predict_spam(message):
    # Preprocess message
    cleaned_text = preprocess_text(message)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad sequence
    sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=100,
        padding="post",
        truncating="post"
    )

    # Make prediction
    prediction = model.predict(sequence, verbose=0)[0][0]

    # Classify message
    if prediction >= 0.5:
        result = "SPAM"
        confidence = prediction
    else:
        result = "NOT SPAM"
        confidence = 1 - prediction

    return result, confidence


# Display project title
print("\n========================================")
print("        SPAM MESSAGE DETECTOR")
print("========================================")

# Keep testing messages until user exits
while True:

    message = input("\nEnter your message (or type 'exit' to quit): ")

    # Exit the program
    if message.lower() == "exit":
        print("\nThank you for using Spam Message Detector!")
        break

    # Predict message
    result, confidence = predict_spam(message)

    # Display result
    print("\n-----------------------------")
    print("Prediction:", result)
    print("Confidence:", round(confidence * 100, 2), "%")
    print("-----------------------------")