import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing  import sequence
from tensorflow.keras.preprocessing import load_model


word_index = imdb.get_word_index()
reverse_word_index = {value : key for key , value in word_index.items()}

model = load_model('simple_rnn_model.h5')


def preprocess_test(text):
    words = text.lower().split()
    sample_review_encoded = [word_index.get(word , 2)+3 for word in words]
    padded_review = sequence.pad_sequences([sample_review_encoded] , maxlen = 500)
    return padded_review

def decoded_review(sample_review_encoded):
    return ' '.join([reverse_word_index.get(i-3 , '?') for i in sample_review_encoded])

# def predict_review(review):
#     preprocessed_input = preprocess_test(review)
#     Y_pred = model.predict(preprocessed_input)
#     sentiment = 'Positive' if Y_pred[0][0] > 0.5 else 'Negative'
#     return sentiment , Y_pred[0][0]
    
    
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_data = preprocess_test(user_input)
    Y_pred = model.predict(preprocess_data)
    sentiment = 'Positive' if Y_pred[0][0] > 0.5 else 'Negative'
    st.write(f'Sentiment :{sentiment}')
    st.write(f'Predcition Score :{Y_pred[0][0]}')
else:
    st.write('Please eneter a movie review')
    