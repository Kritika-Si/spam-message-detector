import tensorflow as tf
import pickle

model = tf.keras.models.load_model("spam_detection_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Test 1
seq1 = tokenizer.texts_to_sequences(["hello"])
seq1 = tf.keras.preprocessing.sequence.pad_sequences(
    seq1,
    maxlen=100,
    padding="post",
    truncating="post"
)

# Test 2
seq2 = tokenizer.texts_to_sequences(
    ["congratulations free prize click link claim reward"]
)
seq2 = tf.keras.preprocessing.sequence.pad_sequences(
    seq2,
    maxlen=100,
    padding="post",
    truncating="post"
)

print("Hello padded sequence:")
print(seq1)

print("\nSpam padded sequence:")
print(seq2)

p1 = model.predict(seq1, verbose=0)[0][0]
p2 = model.predict(seq2, verbose=0)[0][0]

print("\nHello prediction:", p1)
print("Spam prediction:", p2)