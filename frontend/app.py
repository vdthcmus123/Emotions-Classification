import streamlit as st
import requests

st.set_page_config(page_title="Emotions AI Agent", layout="centered")
st.title("Emotion ClassificationAI Chatbot")
st.write("Enter your feelings and let the AI analyze your emotions!")

# Init session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Old conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Receive user input and send to backend for emotion classification
if prompt := st.chat_input("How are you feeling right now?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call API to get emotion prediction
    try:
        response = requests.post("http://localhost:8000/predict", json={"text": prompt})
        if response.status_code == 200:
            result = response.json()
            emotion = result["emotion"]
            score = round(result["confidence"] * 100, 2)
            
            bot_reply = f"I sense you are feeling: {emotion} (Confidence: {score}%)"
            
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    except Exception as e:
        st.error(f"Unable to connect to AI Server: {e}")