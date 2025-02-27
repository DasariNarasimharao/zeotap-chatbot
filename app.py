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
This advanced chatbot helps you with questions about Customer Data Platforms (CDPs):
- Segment
- mParticle
- Lytics
- Zeotap

Now supporting:
- Complex multi-part questions
- Platform comparisons
- Detailed step-by-step answers
- Implementation guidance
""")

# Example questions section with categories
with st.expander("📝 Example Questions", expanded=False):
    st.markdown("### Simple Questions")
    simple_questions = [q for q in get_example_questions()[:4]]
    for q in simple_questions:
        st.markdown(f"- {q}")

    st.markdown("### Complex Questions")
    complex_questions = [q for q in get_example_questions()[4:]]
    for q in complex_questions:
        st.markdown(f"- {q}")

# Platform selector
selected_platform = st.selectbox(
    "Select a specific CDP platform (optional):",
    ["All Platforms", "Segment", "mParticle", "Lytics", "Zeotap"]
)

# Question type hints
question_type = st.radio(
    "What type of question are you asking?",
    ["How-to Guide", "Platform Comparison", "Troubleshooting", "General Information"],
    help="This helps me provide more relevant answers"
)

# Chat input
user_question = st.text_input(
    "Ask your question:",
    key="user_input",
    help="You can ask complex questions or break them down into parts"
)

if st.button("Send"):
    if user_question:
        # Analyze and validate question
        question_analysis = st.session_state.question_handler.analyze_question(user_question)
        is_valid, error_message = st.session_state.question_handler.validate_question(user_question)

        if not is_valid:
            st.error(error_message)
        else:
            # Process question
            platform = None if selected_platform == "All Platforms" else selected_platform.lower()

            # Generate response
            response, similarity = st.session_state.doc_processor.generate_response(
                user_question,
                question_analysis
            )

            # Format response
            formatted_response = format_response(
                response,
                similarity,
                question_analysis,
                platform
            )

            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "response": formatted_response,
                "analysis": question_analysis
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

    # Display analysis for complex questions
    analysis = chat.get('analysis', {})
    analysis_text = ""
    if analysis and analysis.get('complexity') in ['moderate', 'complex']:
        question_types = ', '.join(analysis.get('question_types', []))
        platforms = ', '.join(p.title() for p in analysis.get('platforms', []))
        analysis_text = f"""
        <div class="question-analysis">
            <small>
                Question Type: {question_types}<br>
                Platforms: {platforms or 'General CDP'}<br>
                Complexity: {analysis.get('complexity', 'simple').title()}
            </small>
        </div>
        """

    st.markdown(
        f"""<div class="chat-message bot-message">
            <div class="message-content">{chat['response']['response']}</div>
            {analysis_text}
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