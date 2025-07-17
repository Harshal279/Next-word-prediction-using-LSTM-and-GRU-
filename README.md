# 🔮 Next Word Predictor using LSTM & GRU

[![Streamlit App](https://img.shields.io/badge/Live-App-success?logo=streamlit)](https://kfjhvh2skgdpthmrnitgdc.streamlit.app/)
[![Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Model-Keras-red?logo=keras)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Predict the next word in a sentence using both LSTM and GRU-based models — in real time!

---

### 🚀 Try it Live
👉 **[Click here to open the app](https://kfjhvh2skgdpthmrnitgdc.streamlit.app/)**  
Enter a partial sentence and get next-word predictions from both LSTM and GRU models side by side!

---

### 🧠 Models Used
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

Both models are trained using a custom dataset and saved using Keras:
```python
model.save('next_word_lstm.h5')  # for LSTM
model.save('next_word_gru.h5')   # for GRU
