
# 🛡️ Spam Message Detector

A Machine Learning based web application that detects whether a text message is **Spam** or **Not Spam** using **Python, NLP, TensorFlow (LSTM)** and **Streamlit**.

## 🚀 Live Demo

**🌐 Streamlit App:**  
https://spam-message-detector-8dnzsgyhxdxrdynyjnzd3k.streamlit.app

**💻 GitHub Repository:**  
https://github.com/Kritika-Si/spam-message-detector

---

## 📌 Project Overview

Spam messages are one of the most common forms of unwanted communication. This project uses **Natural Language Processing (NLP)** and a **Long Short-Term Memory (LSTM)** deep learning model to classify SMS messages as either:

- 🚨 **Spam**
- ✅ **Not Spam**

The application provides real-time predictions through an interactive Streamlit web interface.

---

## ✨ Features

- 🔍 Real-time spam message prediction
- 🤖 LSTM Deep Learning model
- 📝 NLP text preprocessing
- 🌐 Interactive Streamlit web app
- 💾 Pre-trained model for instant predictions
- 📱 Clean and responsive dark UI

---

## 🖼️ Application Screenshots

### Home Page

![Home Page](Screenshot%202026-09-23%20211832.png)

### Prediction Interface

![Prediction Interface](Screenshot%202026-09-23%20211853.png)

---

## 🧠 Machine Learning Workflow

1. Input SMS message
2. Text preprocessing using NLP
3. Tokenization & padding
4. LSTM model prediction
5. Display Spam / Not Spam result

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow / Keras | LSTM Deep Learning Model |
| NLP | Text preprocessing |
| NLTK | Stopword removal & tokenization |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing |
| Streamlit | Web application |
| Git & GitHub | Version control |

---

## 📂 Project Structure

```text
spam-message-detector/
│
├── app.py                       # Streamlit application
├── predict.py                   # Prediction logic
├── spam_detection_model.keras   # Trained LSTM model
├── tokenizer.pkl               # Saved tokenizer
├── requirements.txt
├── pyproject.toml
├── README.md
├── Screenshot 2026-09-23 211832.png
└── Screenshot 2026-09-23 211853.png
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Kritika-Si/spam-message-detector.git
```

Move into the project folder:

```bash
cd spam-message-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🧪 Sample Test Messages

| Message | Expected Output |
|---------|----------------|
| Congratulations! You won ₹50,000. Click here now! | 🚨 Spam |
| URGENT! Claim your free prize today. | 🚨 Spam |
| Hey, are we meeting at 5 PM? | ✅ Not Spam |
| Happy Birthday! Have a wonderful day. | ✅ Not Spam |

---

## 📊 Model Information

- **Model:** LSTM (Long Short-Term Memory)
- **Framework:** TensorFlow / Keras
- **Task:** Binary Text Classification
- **Classes:** Spam & Not Spam

---

## 👩‍💻 Author

**Kritika Kumari**

B.Tech CSE (AI & DS)

- GitHub: https://github.com/Kritika-Si
- Live App: https://spam-message-detector-8dnzsgyhxdxrdynyjnzd3k.streamlit.app

---

## ⭐ If you like this project

Give this repository a **Star ⭐** on GitHub and feel free to explore the live application!
