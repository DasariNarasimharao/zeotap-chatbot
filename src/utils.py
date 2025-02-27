def format_response(document, similarity_score, platform=None):
    """Format the response with relevant information and confidence"""
    if similarity_score < 0.2:
        return {
            "response": "I'm sorry, I couldn't find a relevant answer to your question. Please try rephrasing or ask about a specific CDP feature.",
            "confidence": "low",
            "platform": platform
        }
    
    confidence = "high" if similarity_score > 0.7 else "medium" if similarity_score > 0.4 else "low"
    
    return {
        "response": document,
        "confidence": confidence,
        "platform": platform
    }

def get_example_questions():
    """Return a list of example questions"""
    return [
        "How do I set up a new source in Segment?",
        "How can I create a user profile in mParticle?",
        "How do I build an audience segment in Lytics?",
        "How can I integrate my data with Zeotap?",
        "How does Segment's audience creation compare to Lytics?"
    ]
