import streamlit as st
import requests

st.set_page_config(
    page_title="Emotion AI Chatbot",
    page_icon="🎭",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .sidebar-content {
        font-size: 14px;
        color: #4b5563;
    }
    </style>
""", unsafe_allow_html=True)

EMOTION_MAP = {
    "sadness": {"icon": "😢", "color": "blue", "label": "Sad"},
    "joy": {"icon": "😊", "color": "yellow", "label": "Joy"},
    "love": {"icon": "❤️", "color": "red", "label": "Love"},
    "anger": {"icon": "😡", "color": "orange", "label": "Anger"},
    "fear": {"icon": "😨", "color": "purple", "label": "Fear"},
    "surprise": {"icon": "😯", "color": "green", "label": "Surprise"}
}

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

#  SIDEBAR  
with st.sidebar:
    st.title("Chat History")
    st.markdown("---")
    if not st.session_state.messages:
        st.info("No messages yet.")
    else:
        user_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        for i, query in enumerate(reversed(user_queries)):
            st.markdown(f"**{i+1}.** {query[:30]}..." if len(query) > 30 else f"**{i+1}.** {query}")
    
    st.markdown("---")
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# UI
st.title("Emotion Classification AI")
st.caption("Enter a sentence to analyze the emotion behind it. Powered by a fine-tuned BERT model.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How do you feel today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call API Backend
    try:
        with st.spinner("Analyzing emotion..."):
            response = requests.post("http://localhost:8000/predict", json={"text": prompt})
            
        if response.status_code == 200:
            result = response.json()
            emotion = result["emotion"]
            score = result["confidence"]
            
            emo_info = EMOTION_MAP.get(emotion, {"icon": "🤔", "color": "gray", "label": emotion})
            
            bot_response = f"""
            According to the analysis, your emotion is: **{emo_info['label']}** {emo_info['icon']}
            
            * **Confidence:** `{score:.2%}`
            * **Context Analysis:** The BERT model detects signs of {emo_info['label'].lower()} in your statement.
            """
            
            with st.chat_message("assistant"):
                st.markdown(bot_response)
                st.progress(score, text=f"Confidence Score: {score:.2%}")
                
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
        else:
            st.error("Error: Server did not respond with the expected format.")
    except Exception as e:
        st.error(f"Error: Cannot connect to backend: {e}")
