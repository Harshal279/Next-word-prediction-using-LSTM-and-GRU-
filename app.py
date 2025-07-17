import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM and GRU models
lstm_model = load_model('next_word_lstm.h5')
gru_model = load_model('next_word_gru.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted_probs, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"

# Streamlit UI
st.title("Next Word Prediction with LSTM & GRU")
st.write("Enter a sequence of words and compare next-word predictions from both models.")

input_text = st.text_input("Enter the sequence of words:", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_len = lstm_model.input_shape[1] + 1  # assuming both models have same input length

    lstm_prediction = predict_next_word(lstm_model, tokenizer, input_text, max_sequence_len)
    gru_prediction = predict_next_word(gru_model, tokenizer, input_text, max_sequence_len)

    st.markdown("### 🔮 Predictions")
    st.write(f"**LSTM Prediction:** `{lstm_prediction}`")
    st.write(f"**GRU Prediction:** `{gru_prediction}`")

    if lstm_prediction == gru_prediction:
        st.success("Both models agree on the prediction!")
    else:
        st.info("The models predicted different next words.")
