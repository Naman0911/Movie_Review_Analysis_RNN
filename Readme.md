# 🎬 Movie Review Sentiment Analysis using Simple RNN

A Deep Learning based Sentiment Analysis project that classifies IMDB movie reviews as **Positive** or **Negative** using a **Simple Recurrent Neural Network (RNN)** built with TensorFlow and Keras.

The project also includes an interactive **Streamlit Web Application** for real-time sentiment prediction.

---

## 🚀 Live Demo

🔗 Streamlit App:  
https://moviereviewanalysisrnn-drwqc2eap7k8pngfmbt56v.streamlit.app/

---

## 🚀 Features

- Movie Review Sentiment Classification
- Deep Learning based NLP pipeline
- Simple RNN Architecture
- IMDB Dataset Integration
- Interactive Streamlit Frontend
- Real-time Prediction
- Clean and Modern UI
- Pre-trained Model Support

---

## 🧠 Model Architecture

The model is built using:

```python
model = Sequential([
    Input(shape=(500,)),
    Embedding(10000, 128),
    SimpleRNN(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Training Accuracy | 90% |
| Task | Binary Classification |
| Dataset | IMDB Movie Reviews |

---

## 📁 Project Structure

```bash
Movie_Review_Analysis_RNN/
│
├── app.py
├── simple_rnn.ipynb
├── prediction.ipynb
├── simple_rnn_model.h5
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Streamlit
- IMDB Dataset

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Naman0911/Movie_Review_Analysis_RNN.git
cd Movie_Review_Analysis_RNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open in browser:

```bash
http://localhost:8501
```

---

## 💡 How It Works

1. User enters a movie review
2. Text gets tokenized and padded
3. Review is passed through the trained RNN model
4. Model predicts sentiment probability
5. Final output is shown as Positive 😊 or Negative 😔

---

## 🧪 Dataset

The project uses the IMDB Movie Reviews Dataset available in TensorFlow/Keras datasets.

- 50,000 movie reviews
- Binary sentiment labels
- Preprocessed integer encoded sequences

---

## 🔮 Future Improvements

- Upgrade to LSTM/GRU
- Add Attention Mechanism
- Deploy on Streamlit Cloud
- Use Pretrained Word Embeddings
- Add Confidence Visualization

---

## 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository and submit pull requests.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed by **Naman Upadhyay**

GitHub: https://github.com/Naman0911
