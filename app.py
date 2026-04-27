import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Chatbot RAG",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* Header */
.chat-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.5rem;
}

.chat-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    color: #60a5fa;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}

.chat-header p {
    color: #64748b;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Messages */
.user-msg {
    background: #1e3a5f;
    border: 1px solid #2563eb33;
    border-radius: 12px 12px 2px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #e2e8f0;
}

.bot-msg {
    background: #131820;
    border: 1px solid #1e2535;
    border-radius: 2px 12px 12px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #cbd5e1;
}

.sources-block {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 0.6rem 1rem;
    margin: 0.25rem 3rem 0.75rem 0;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #60a5fa;
    white-space: pre-wrap;
}

.role-label {
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #475569;
    margin-bottom: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Input area */
.stTextInput input {
    background: #131820 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px #2563eb22 !important;
}

/* Buttons */
.stButton button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: background 0.2s !important;
}

.stButton button:hover {
    background: #1d4ed8 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0f14 !important;
    border-right: 1px solid #1e2535 !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Spinner */
.stSpinner > div {
    border-top-color: #2563eb !important;
}
</style>
""", unsafe_allow_html=True)


try:
    from agent.chatbot_graph import ask_chatbot
    BACKEND_LOADED = True
except Exception as e:
    BACKEND_LOADED = False
    LOAD_ERROR = str(e)

st.markdown("""
<div class="chat-header">
    <h1>⬡ RAG Chatbot</h1>
    <p>Powered by GPT-4o · LangGraph · ChromaDB</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configuration")
    
    if not BACKEND_LOADED:
        st.error(f"Erreur de chargement : {LOAD_ERROR}")
    else:
        st.success("Backend connecté")
    
    st.markdown("---")
    
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#475569; font-family:'IBM Plex Mono',monospace;">
    Ce chatbot répond aux questions<br>
    à partir de vos documents internes.<br><br>
    Formats supportés :<br>
    📄 PDF · 📊 Excel
    </div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="role-label">Vous</div><div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="role-label">Assistant</div><div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            st.markdown(f'<div class="sources-block">{msg["sources"]}</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="bot-msg" style="border-color:#1e3a5f; color:#94a3b8; font-style:italic;">
    Bonjour ! Posez-moi une question sur vos documents internes.<br>
    Je cite toujours mes sources.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="Question",
            placeholder="Posez votre question...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Envoyer")

if submitted and user_input.strip():
    question = user_input.strip()
    
    st.session_state.messages.append({"role": "user", "content": question})
    
    if not BACKEND_LOADED:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Le backend n'est pas chargé correctement.",
            "sources": None
        })
    else:
        with st.spinner("Recherche dans les documents..."):
            try:
                result = ask_chatbot(question, st.session_state.history)
                
                # Mise à jour de l'historique LangGraph
                st.session_state.history.append(HumanMessage(content=question))
                st.session_state.history.append(AIMessage(content=result["response"]))
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "sources": result.get("sources")
                })
                
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Erreur : {str(e)}",
                    "sources": None
                })
    
    st.rerun()