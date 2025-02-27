import re

class QuestionHandler:
    def __init__(self):
        self.cdp_keywords = {
            'segment': ['segment', 'segments'],
            'mparticle': ['mparticle', 'mparticles'],
            'lytics': ['lytics'],
            'zeotap': ['zeotap']
        }
        
    def identify_platform(self, question):
        """Identify which CDP platform the question is about"""
        question = question.lower()
        for platform, keywords in self.cdp_keywords.items():
            if any(keyword in question for keyword in keywords):
                return platform
        return None
    
    def is_how_to_question(self, question):
        """Check if the question is a how-to question"""
        question = question.lower()
        how_to_patterns = [
            r'^how\s+(?:do|can|to|would|should)',
            r'^what(?:\s+is)?(?:\s+the)?\s+(?:way|process|method)',
            r'steps?\s+to',
            r'guide\s+(?:me|us)?\s+(?:on|through)',
        ]
        return any(re.search(pattern, question) for pattern in how_to_patterns)
    
    def is_comparison_question(self, question):
        """Check if the question is asking for a comparison"""
        question = question.lower()
        comparison_patterns = [
            r'compare',
            r'difference',
            r'versus',
            r'vs',
            r'better',
            r'similar to'
        ]
        return any(pattern in question for pattern in comparison_patterns)
    
    def validate_question(self, question):
        """Validate if the question is relevant to CDPs"""
        if not question.strip():
            return False, "Please enter a question."
        
        if len(question) > 500:
            return False, "Question is too long. Please be more specific."
            
        if not self.is_how_to_question(question) and not self.is_comparison_question(question):
            return False, "Please ask a 'how-to' question or comparison question about CDP platforms."
            
        return True, None
