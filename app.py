import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

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

    return y


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.write("### Detect whether a message is **Spam** or **Not Spam** using Machine Learning.")

input_sms = st.text_area("Enter your message below:", height=150, placeholder="Type your message here...")

col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    predict_clicked = st.button("Predict")
if predict_clicked:
    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        transformed_sms = " ".join(transformed_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("This message is Spam!")
        else:
            st.success("This message is Not Spam.")

