# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import com.thetopon.vector.rag.app as ap

load_dotenv()

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

st.header("Settings")
website_url = "https://www.energyplus.qa/pressurized_habitat.php"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"), ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = ap.get_vectorstore_from_url(website_url)

    # user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = ap.get_response(user_query, st.session_state.vector_store, st.session_state.chat_history)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

