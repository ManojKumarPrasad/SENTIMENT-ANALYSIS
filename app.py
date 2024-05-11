import streamlit as st
import pickle
import nltk
nltk.download("stopwords")

import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

model = pickle.load(open("modell.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Movie Review Sentiment Analysis")
input_text = st.text_area('Enter the message here')

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Negation handling
    text = re.sub(r'\bnot\b\s+(.+)', r'not_\1', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_text = ' '.join([stemmer.stem(word) for word in text.split()])

    return stemmed_text


if st.button("Predict"):
    transformed_text = preprocess_text(input_text)

    vector_input = tfidf.transform([transformed_text])

    result = model.predict(vector_input)

    if result == 1:
        st.header("Positive")
    else:
        st.header("Negative")

# git init
# git add README.md
# git commit -m "first commit"
# git branch -M main
# git remote add origin https://github.com/ManojKumarPrasad/SENTIMENT-ANALYSIS.git
# git push -u origin main