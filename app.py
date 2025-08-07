import streamlit as st
from voice_module import listen
from nlp_module import get_answer
from retriever import retrieve_context

st.set_page_config(page_title="Bhuvan Voice Chatbot", page_icon="🛰️")

st.title("🛰️ Bhuvan Voice Chatbot")
st.write("Ask questions about any Bhuvan applications using your voice or text.")

# Session state for context
if 'context' not in st.session_state:
    st.session_state.context = ""

# Voice input
if st.button("🎙️ Speak Your Question"):
    query = listen()
    st.text_input("You said:", query, key="voice_input")

# Text input fallback
query = st.text_input("Or type your question here:")

if query:
    context = retrieve_context(query)
    answer = get_answer(query, context)
    st.session_state.context = context
    st.markdown(f"**Answer:** {answer}")

