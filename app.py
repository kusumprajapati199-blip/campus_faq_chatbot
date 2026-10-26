import streamlit as st
import pandas as pd
from responsible_ai import ResponsibleChatbotAgent

# Page configuration
st.set_page_config(
    page_title="Campus Service FAQ Chatbot",
    page_icon="🎓",
    layout="wide"
)

# Cache agent instance so embeddings load only once
@st.cache_resource
def load_agent():
    return ResponsibleChatbotAgent(data_path="data/campus_faqs.json", confidence_threshold=0.45)

agent = load_agent()

# Sidebar: Controls and Methodology Settings
st.sidebar.title("⚙️ AI Configuration")
st.sidebar.markdown("Compare retrieval techniques in real-time.")

selected_method = st.sidebar.selectbox(
    "Choose Retrieval Engine:",
    ("Sentence-Transformers", "TF-IDF", "Keyword (Jaccard)")
)

threshold = st.sidebar.slider(
    "Confidence Threshold (Guardrail)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    step=0.05
)
agent.confidence_threshold = threshold

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Categories")
st.sidebar.markdown("- 🏠 Hostel\n- 📚 Library\n- 💳 Fees\n- 🪪 ID Cards\n- 🚌 Transport\n- 📝 Examinations")

# Main Page Header
st.title("🎓 Campus Service Chatbot")
st.caption(f"Active Retrieval Engine: **{selected_method}** | Confidence Guardrail: **{threshold}**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            with st.expander("🔍 View AI Reasoning & Explainability"):
                st.json(msg["metadata"])

# Input for new user query
if prompt := st.chat_input("Ask a campus question (e.g., 'How to reissue lost ID card?', 'Bus pass procedure')..."):
    # Append & display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query through Responsible AI pipeline
    response_data = agent.process_query(prompt, selected_method=selected_method)

    # Format assistant answer
    answer_text = response_data["answer"]
    meta_info = {
        "Status": response_data["status"],
        "Confidence Score": response_data["confidence_score"],
        "Matched FAQ": response_data.get("matched_question"),
        "Category": response_data.get("category"),
        "Explainability Trace": response_data["explainability"]
    }

    # Display assistant response
    with st.chat_message("assistant"):
        if response_data["status"] == "FALLBACK":
            st.warning(answer_text)
        else:
            st.success(answer_text)

        with st.expander("🔍 View AI Reasoning & Explainability"):
            st.json(meta_info)

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "metadata": meta_info
    })
