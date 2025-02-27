from typing import Dict, Any, List

def format_response(response: str, similarity_score: float, question_analysis: Dict[str, Any] = None, platform: str = None) -> Dict[str, Any]:
    """Enhanced response formatting with detailed analysis"""

    # Base confidence thresholds
    confidence_thresholds = {
        'high': 0.7,
        'medium': 0.4,
        'low': 0.2
    }

    # Adjust confidence based on question complexity
    if question_analysis and question_analysis.get('complexity') == 'complex':
        confidence_thresholds = {
            'high': 0.6,  # Lower threshold for complex questions
            'medium': 0.3,
            'low': 0.1
        }

    # Determine confidence level
    if similarity_score < confidence_thresholds['low']:
        return {
            "response": "I apologize, but I couldn't find a sufficiently relevant answer to your question. Could you please:\n"
                       "1. Be more specific about which CDP platform you're asking about\n"
                       "2. Break down complex questions into simpler parts\n"
                       "3. Ensure your question is about CDP features or functionality",
            "confidence": "low",
            "platform": platform,
            "analysis": question_analysis
        }

    confidence = (
        "high" if similarity_score > confidence_thresholds['high']
        else "medium" if similarity_score > confidence_thresholds['medium']
        else "low"
    )

    # Format response based on question type
    formatted_response = response
    if question_analysis:
        if 'comparison' in question_analysis['question_types']:
            formatted_response = f"Comparison Analysis:\n{response}"
        elif question_analysis['complexity'] == 'complex':
            if not response.startswith("Step-by-step"):
                formatted_response = f"Detailed Answer:\n{response}"

    return {
        "response": formatted_response,
        "confidence": confidence,
        "platform": platform,
        "analysis": question_analysis
    }

def get_example_questions() -> List[str]:
    """Return an expanded list of example questions"""
    return [
        # Simple questions
        "How do I set up a new source in Segment?",
        "How can I create a user profile in mParticle?",
        "How do I build an audience segment in Lytics?",
        "How can I integrate my data with Zeotap?",

        # Complex questions
        "How does Segment's audience creation compare to Lytics?",
        "What are the steps to implement user tracking in Segment and validate the data flow?",
        "Can you explain the difference between mParticle and Segment for mobile app tracking?",
        "How can I troubleshoot failed data ingestion in Zeotap and verify the data quality?"
    ]

def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata for display"""
    if not metadata:
        return ""

    formatted = []
    for key, value in metadata.items():
        formatted.append(f"{key.replace('_', ' ').title()}: {value}")

    return "\n".join(formatted)