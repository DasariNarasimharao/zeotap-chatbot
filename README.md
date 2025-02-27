# CDP Support Chatbot

An advanced Streamlit-based chatbot that provides comprehensive answers to CDP (Customer Data Platform) related questions. The chatbot supports multiple CDP platforms including Segment, mParticle, Lytics, and Zeotap.

## Features

- **Multi-Platform Support**: Handles questions about Segment, mParticle, Lytics, and Zeotap
- **Advanced Question Processing**: 
  - Complex multi-part questions
  - Platform comparisons
  - Detailed step-by-step answers
- **Smart Response Generation**:
  - Enhanced similarity matching
  - Context-aware responses
  - Confidence scoring
- **User-Friendly Interface**:
  - Interactive chat interface
  - Example questions
  - Platform selection
  - Question type hints

## Technologies Used

- Streamlit for web interface
- Advanced NLP with enhanced query processing
- TF-IDF and BM25 for similarity matching
- NLTK for text processing

## Setup Instructions

1. Install dependencies:
```bash
pip install streamlit nltk scikit-learn numpy rank-bm25
```

2. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Main Streamlit application
├── src/
│   ├── document_processor.py   # Document processing and matching
│   ├── question_handler.py     # Question analysis and validation
│   └── utils.py               # Utility functions
├── assets/
│   └── style.css             # Custom styling
└── .streamlit/
    └── config.toml           # Streamlit configuration
```

## Usage

1. Select a specific CDP platform (optional)
2. Choose the type of question you're asking
3. Enter your question in the chat input
4. View the response with confidence level and analysis

## Example Questions

### Simple Questions
- "How do I set up a new source in Segment?"
- "How can I create a user profile in mParticle?"
- "How do I build an audience segment in Lytics?"
- "How can I integrate my data with Zeotap?"

### Complex Questions
- "How does Segment's audience creation compare to Lytics?"
- "What are the steps to implement user tracking in Segment and validate the data flow?"
- "Can you explain the difference between mParticle and Segment for mobile app tracking?"
