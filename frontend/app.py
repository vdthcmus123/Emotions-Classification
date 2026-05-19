import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
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
    "sadness": {"icon": "😢", "color": "blue",   "label": "Sad"},
    "joy":     {"icon": "😊", "color": "yellow",  "label": "Joy"},
    "love":    {"icon": "❤️", "color": "red",     "label": "Love"},
    "anger":   {"icon": "😡", "color": "orange",  "label": "Anger"},
    "fear":    {"icon": "😨", "color": "purple",  "label": "Fear"},
    "surprise":{"icon": "😯", "color": "green",   "label": "Surprise"},
}

# Session State
def _new_conversation():
    """Append a blank conversation and make it active."""
    st.session_state.conversations.append({"title": "New Conversation", "messages": []})
    st.session_state.active_idx = len(st.session_state.conversations) - 1

if "conversations" not in st.session_state:
    st.session_state.conversations = []
    _new_conversation()          # always start with one open conversation

if "active_idx" not in st.session_state:
    st.session_state.active_idx = 0

# Convenience alias (re-evaluated each run)
active = st.session_state.conversations[st.session_state.active_idx]

# SIDEBAR 
with st.sidebar:
    st.title("Chat History")

    if st.button("＋  New Conversation", use_container_width=True, type="primary"):
        _new_conversation()
        st.rerun()

    st.markdown("---")

    if len(st.session_state.conversations) == 0:
        st.info("No conversations yet.")
    else:
        # Show most-recent first
        for i in range(len(st.session_state.conversations) - 1, -1, -1):
            conv = st.session_state.conversations[i]
            label = conv["title"]
            # Highlight the active conversation
            is_active = i == st.session_state.active_idx
            btn_label = f"▶ {label}" if is_active else label
            if st.button(btn_label, key=f"conv_{i}", use_container_width=True):
                st.session_state.active_idx = i
                st.rerun()

    st.markdown("---")
    if st.button("🗑  Clear All History", use_container_width=True):
        st.session_state.conversations = []
        _new_conversation()
        st.rerun()

# Re-bind after potential rerun guards
active = st.session_state.conversations[st.session_state.active_idx]

# MAIN UI
st.title("Emotion Classification AI")
st.caption("Enter a sentence to analyze the emotion behind it. Powered by a fine-tuned BERT model.")

# Render existing messages for the active conversation
for message in active["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How do you feel today?"):
    # Set conversation title from the very first user message
    if not active["messages"]:
        active["title"] = prompt[:40] + ("…" if len(prompt) > 40 else "")

    st.chat_message("user").markdown(prompt)
    active["messages"].append({"role": "user", "content": prompt})

    # Call API Backend
    try:
        with st.spinner("Analyzing emotion..."):
            response = requests.post(f"{BACKEND_URL}/predict", json={"text": prompt})

        if response.status_code == 200:
            result   = response.json()
            emotion  = result["emotion"]
            score    = result["confidence"]
            all_probs = result["all_probs"]
            
            emo_info = EMOTION_MAP.get(emotion, {"icon": "🤔", "color": "gray", "label": emotion})

            bot_response = (
                f"According to the analysis, your emotion is: **{emo_info['label']}** {emo_info['icon']}\n\n"
                f"* **Confidence:** `{score:.2%}`\n"
                f"* **Context Analysis:** The BERT model detects signs of "
                f"{emo_info['label'].lower()} in your statement."
            )

            with st.chat_message("assistant"):
                st.markdown(bot_response)
                st.progress(score, text=f"Confidence Score: {score:.2%}")
                st.markdown("**Biểu đồ phân bổ cảm xúc:**")
                st.bar_chart(all_probs)

            active["messages"].append({"role": "assistant", "content": bot_response})

        else:
            st.error("Error: Server did not respond with the expected format.")

    except Exception as e:
        st.error(f"Error: Cannot connect to backend: {e}")