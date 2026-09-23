import streamlit as st
import tensorflow as tf
import pickle
import string
import nltk
from nltk.corpus import stopwords


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📧",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# DOWNLOAD STOPWORDS
# ==========================================

nltk.download("stopwords", quiet=True)

stop_words = stopwords.words("english")


# ==========================================
# LOAD MODEL AND TOKENIZER
# ==========================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model(
        "spam_detection_model.keras"
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_resources()


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = []

    for word in text.split():

        word = word.lower()

        if word not in stop_words:
            words.append(word)

    return " ".join(words)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_spam(message):

    cleaned_text = preprocess_text(message)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=100,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )[0][0]

    if prediction >= 0.5:

        result = "SPAM"
        confidence = prediction

    else:

        result = "NOT SPAM"
        confidence = 1 - prediction

    return result, confidence


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">📧 Spam Message Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect whether a message is <b>Spam</b> or <b>Not Spam</b> using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# MESSAGE INPUT
# ==========================================

st.subheader("📝 Enter Your Message")

message = st.text_area(
    "Message",
    height=160,
    placeholder=(
        "Example: Congratulations! "
        "You have won a free prize. Click here to claim!"
    ),
    label_visibility="collapsed"
)


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(
    "🔍 Check Message",
    use_container_width=True
):

    if message.strip() == "":

        st.warning(
            "⚠️ Please enter a message before checking."
        )

    else:

        result, confidence = predict_spam(message)

        confidence_percent = confidence * 100


        # ==================================
        # SPAM RESULT
        # ==================================

        if result == "SPAM":

            st.error(
                f"🚨 SPAM MESSAGE"
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <h2>🚨 SPAM</h2>
                    <h3>Confidence: {confidence_percent:.2f}%</h3>
                    <p>This message has characteristics commonly associated with spam.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ==================================
        # NOT SPAM RESULT
        # ==================================

        else:

            st.success(
                f"✅ NOT SPAM"
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <h2>✅ NOT SPAM</h2>
                    <h3>Confidence: {confidence_percent:.2f}%</h3>
                    <p>This message appears to be a normal message.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.divider()

st.subheader("🤖 About This Project")

st.write(
    "This application uses a trained LSTM "
    "(Long Short-Term Memory) neural network "
    "to classify text messages as Spam or Not Spam."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "LSTM")

with col2:
    st.metric("Sequence Length", "100")

with col3:
    st.metric("Test Accuracy", "99.17%")


# ==========================================
# HOW IT WORKS
# ==========================================

with st.expander("🔎 How does it work?"):

    st.write(
        """
        1. The user enters a message.
        
        2. The text is cleaned by removing punctuation
           and English stopwords.
        
        3. The tokenizer converts the text into numerical
           sequences.
        
        4. The sequence is padded to a fixed length of 100.
        
        5. The trained LSTM model analyzes the message.
        
        6. The application displays the predicted class
           and confidence.
        """
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Spam Message Detector | Machine Learning Project"
)