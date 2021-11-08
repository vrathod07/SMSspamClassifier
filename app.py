import pickle

import nltk
import streamlit as st
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))


st.title('SMS Spam Classifier')
input_message = st.text_input('Enter the message/SMS')

if st.button('Predict'):
  #1. Preprocess
  ps = PorterStemmer()
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

  transformed_sms = transform_text(input_message)

  #2. Vectorize
  vec_input = tfidf.transform([transformed_sms])

  #3. predict
  result = model.predict(vec_input)[0]

  #4. Display
  if result == 0:
    st.header('NOT SPAM')
    print('ham')
  else:
    st.header('SPAM')
    print('spam')