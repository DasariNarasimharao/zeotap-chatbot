import streamlit as st
from src.document_processor import DocumentProcessor
from src.question_handler import QuestionHandler
from src.utils import format_response, get_example_questions

# Page configuration
st.set_page_config(
    page_title="CDP Support Chatbot",
    page_icon="💬",
    layout="wide"
)

# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

if 'question_handler' not in st.session_state:
    st.session_state.question_handler = QuestionHandler()

# Main layout
st.title("CDP Support Assistant 💬")
st.markdown("""
This chatbot helps you with questions about Customer Data Platforms (CDPs):
- Segment
- mParticle
- Lytics
- Zeotap
""")

# Example questions section
with st.expander("📝 Example Questions"):
    for question in get_example_questions():
        st.markdown(f"- {question}")

# Platform selector
selected_platform = st.selectbox(
    "Select a specific CDP platform (optional):",
    ["All Platforms", "Segment", "mParticle", "Lytics", "Zeotap"]
)

# Chat input
user_question = st.text_input("Ask your question:", key="user_input")

if st.button("Send"):
    if user_question:
        # Validate question
        is_valid, error_message = st.session_state.question_handler.validate_question(user_question)

        if not is_valid:
            st.error(error_message)
        else:
            # Process question
            platform = None if selected_platform == "All Platforms" else selected_platform.lower()

            # Get response
            doc, similarity = st.session_state.doc_processor.find_relevant_document(
                user_question,
                platform
            )

            # Format response
            response = format_response(doc, similarity, platform)

            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "response": response
            })

# Display chat history
for chat in st.session_state.chat_history:
    # User message
    st.markdown(
        f"""<div class="chat-message user-message">
            <div class="message-content">{chat['question']}</div>
        </div>""",
        unsafe_allow_html=True
    )

    # Bot response
    confidence_color = {
        "high": "green",
        "medium": "orange",
        "low": "red"
    }[chat['response']['confidence']]

    platform_info = f"Platform: {chat['response']['platform'].title()}" if chat['response']['platform'] else ""

    st.markdown(
        f"""<div class="chat-message bot-message">
            <div class="message-content">{chat['response']['response']}</div>
            <div class="confidence-indicator" style="color: {confidence_color}">
                Confidence: {chat['response']['confidence'].title()} {platform_info}
            </div>
        </div>""",
        unsafe_allow_html=True
    )

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()