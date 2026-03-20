import streamlit as st
from googletrans import Translator

translator = Translator()

st.set_page_config(page_title="Language Translation Tool")
st.title("Language Translation Tool")

text = st.text_area("Enter text to translate")

languages = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es"
}

source = st.selectbox("Source Language", list(languages.keys()))
target = st.selectbox("Target Language", list(languages.keys()))

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        result = translator.translate(
            text,
            src=languages[source],
            dest=languages[target]
        )
        st.success("Translated Text")
        st.write(result.text)
