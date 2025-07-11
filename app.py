import streamlit as st
from src.retriever import Retriever
from src.generator import Generator

st.set_page_config(page_title="RAG Chatbot", layout="wide")

retriever = Retriever()
generator = Generator()

st.title("🧠 RAG Chatbot – Legal Document Q&A")
st.markdown("Ask any question based on the document you uploaded.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("💬 Your question:", placeholder="e.g., What is the termination policy?")

# On submit
if user_input:
    with st.spinner("Searching and thinking..."):
        top_chunks = retriever.search(user_input, top_k=3)
        answer = generator.generate_answer(top_chunks, user_input)

        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer,
            "sources": top_chunks
        })

# Display chat history
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    with st.expander("📄 Source Chunks"):
        for src in chat['sources']:
            st.markdown(f"**Chunk {src['chunk_id']} (Score: {src['score']:.2f})**")
            st.markdown(f"> {src['text']}")
    st.markdown("---")

# Sidebar info
with st.sidebar:
    st.subheader("📊 Info")
    st.markdown(f"**Model:** Mistral-7B-Instruct (via LM Studio)")
    st.markdown(f"**Chunks Loaded:** {len(retriever.chunk_texts)}")
    st.button("🔁 Reset Chat", on_click=lambda: st.session_state.chat_history.clear())
