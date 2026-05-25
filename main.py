# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing  import sequence
# from tensorflow.keras.models import load_model


# word_index = imdb.get_word_index()
# reverse_word_index = {value : key for key , value in word_index.items()}

# model = load_model('simple_rnn_model.h5')


# def preprocess_test(text):
#     words = text.lower().split()
#     sample_review_encoded = [word_index.get(word , 2)+3 for word in words]
#     padded_review = sequence.pad_sequences([sample_review_encoded] , maxlen = 500)
#     return padded_review

# def decoded_review(sample_review_encoded):
#     return ' '.join([reverse_word_index.get(i-3 , '?') for i in sample_review_encoded])

# # def predict_review(review):
# #     preprocessed_input = preprocess_test(review)
# #     Y_pred = model.predict(preprocessed_input)
# #     sentiment = 'Positive' if Y_pred[0][0] > 0.5 else 'Negative'
# #     return sentiment , Y_pred[0][0]
    
    
# import streamlit as st
# st.title('IMDB Movie Review Sentiment Analysis')
# st.write("Enter a movie review to classify it as positive or negative")

# user_input = st.text_area('Movie Review')

# if st.button('Classify'):
#     preprocess_data = preprocess_test(user_input)
#     Y_pred = model.predict(preprocess_data)
#     sentiment = 'Positive' if Y_pred[0][0] > 0.5 else 'Negative'
#     st.write(f'Sentiment :{sentiment}')
#     st.write(f'Predcition Score :{Y_pred[0][0]}')
# else:
#     st.write('Please eneter a movie review')
    
    
    
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# ----------------------------
# Load IMDB Dictionary + Model
# ----------------------------

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_model.h5')

# ----------------------------
# Preprocessing Function
# ----------------------------

def preprocess_test(text):
    words = text.lower().split()
    sample_review_encoded = [word_index.get(word, 2) + 3 for word in words]

    padded_review = sequence.pad_sequences(
        [sample_review_encoded],
        maxlen=500
    )

    return padded_review


# ----------------------------
# Streamlit Page Config
# ----------------------------

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #BBBBBB;
    margin-bottom: 30px;
}

.stTextArea textarea {
    font-size: 16px;
    border-radius: 10px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.positive {
    background-color: rgba(0,255,0,0.15);
    color: #00FF88;
}

.negative {
    background-color: rgba(255,0,0,0.15);
    color: #FF4B4B;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------

st.markdown(
    '<div class="title">🎬 IMDB Movie Review Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning based Sentiment Classification using Simple RNN</div>',
    unsafe_allow_html=True
)

# ----------------------------
# Input Box
# ----------------------------

user_input = st.text_area(
    "Enter Movie Review",
    height=200,
    placeholder="Type your movie review here..."
)

# ----------------------------
# Predict Button
# ----------------------------

if st.button("Analyze Sentiment 🚀"):

    if user_input.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        preprocess_data = preprocess_test(user_input)

        prediction = model.predict(preprocess_data)

        score = prediction[0][0]

        sentiment = "Positive 😊" if score > 0.5 else "Negative 😔"

        confidence = score if score > 0.5 else 1 - score

        # ----------------------------
        # Result Box
        # ----------------------------

        if score > 0.5:

            st.markdown(
                f'''
                <div class="result-box positive">
                    {sentiment}
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'''
                <div class="result-box negative">
                    {sentiment}
                </div>
                ''',
                unsafe_allow_html=True
            )

        # ----------------------------
        # Confidence Score
        # ----------------------------

        st.write("### Confidence Score")

        st.progress(float(confidence))

        st.write(f"Model Confidence: **{confidence*100:.2f}%**")

        # ----------------------------
        # Raw Prediction
        # ----------------------------

        st.write("### Prediction Probability")

        st.write(f"Sentiment Score: **{score:.4f}**")
