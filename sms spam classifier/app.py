import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

from nltk.corpus import stopwords
stopwords.words('english')
import string

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('mnb.pkl','rb'))

st.title('Sms spam classifer')

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result ==1:
        st.header("spam")
    else:
        st.header("Not spam")













