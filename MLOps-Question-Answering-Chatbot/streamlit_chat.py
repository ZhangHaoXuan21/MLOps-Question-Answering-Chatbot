import requests
import streamlit as st


st.title("MLOps Q/A Chatbot 🤖🏭")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        url = "http://127.0.0.1:8000/chatResponse"  # Replace with your server URL if deployed

        # Define the JSON payload
        payload = {
            "user_query": user_query
        }

        # Make the POST request
        response = requests.post(url, json=payload).json()

        answer = response['response']

        st.markdown(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})



