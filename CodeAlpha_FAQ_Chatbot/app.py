import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import faqs

nltk.download('punkt')

st.set_page_config(page_title="FAQ Chatbot")
st.title("🤖 FAQ Chatbot")

questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

user_question = st.text_input("Ask your question")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question")
    else:
        user_vector = vectorizer.transform([user_question])
        similarity = cosine_similarity(user_vector, question_vectors)
        best_match = similarity.argmax()
        st.success("Answer")
        st.write(answers[best_match])
