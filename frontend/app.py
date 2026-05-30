import streamlit as st
import requests
import os

API_BASE = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api")

st.set_page_config(
    page_title="ChatPDF",
    page_icon="📄",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}

.chat-container {
    max-width: 800px;
    margin: auto;
}

.user-msg {
    background-color: #1f77b4;
    padding: 10px 15px;
    border-radius: 10px;
    color: white;
    margin-bottom: 10px;
}

.bot-msg {
    background-color: #262730;
    padding: 10px 15px;
    border-radius: 10px;
    color: white;
    margin-bottom: 10px;
}

.source-box {
    background-color: #1a1d24;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("📄 Chat with your PDF")
st.caption("Ask questions and get answers directly from your documents")

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            files = {"file": uploaded_file}
            res = requests.post(f"{API_BASE}/upload", files=files)

        if res.status_code == 200:
            st.success("Uploaded & indexed ✅")
        else:
            st.error("Upload failed ❌")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

# ------------------ CHAT AREA ------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

# ------------------ INPUT ------------------
query = st.chat_input("Ask something about your PDF...")

if query:
    # USER MESSAGE
    st.session_state.messages.append({"role": "user", "content": query})

    # Display immediately
    st.markdown(f'<div class="user-msg">👤 {query}</div>', unsafe_allow_html=True)

    # CALL BACKEND
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_BASE}/query",
                json={"question": query}
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer")
                sources = data.get("sources", [])

                # Display answer
                st.markdown(f'<div class="bot-msg">🤖 {answer}</div>', unsafe_allow_html=True)

                # Display sources
                if sources:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources):
                            score = src.get("metadata").get("relevance_score")
                            score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                            st.markdown(
                                f"""
                                <div class="source-box">
                                <b>Source {i+1}</b> (Score: {score_text})<br>
                                {src.get('content')[:300]}...
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            else:
                st.error("Server error")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)