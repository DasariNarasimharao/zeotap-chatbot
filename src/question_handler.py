import re
from typing import Tuple, List, Optional

class QuestionHandler:
    def __init__(self):
        self.cdp_keywords = {
            'segment': ['segment', 'segments', 'segmentation'],
            'mparticle': ['mparticle', 'mparticles', 'particle'],
            'lytics': ['lytics', 'lytic'],
            'zeotap': ['zeotap']
        }

        self.question_types = {
            'implementation': [r'how\s+(?:do|can|to|would|should)', r'steps?\s+to', r'guide\s+(?:me|us)?\s+(?:on|through)'],
            'comparison': [r'(?:compare|versus|vs|difference|better|similar)', r'which\s+(?:is|are)\s+better'],
            'integration': [r'integrate|connection|connect|setup|configure'],
            'troubleshooting': [r'(?:fix|solve|resolve|debug|error|issue|problem)', r'not\s+working', r'fails?\s+to'],
            'conceptual': [r'what\s+(?:is|are|does)', r'explain', r'define', r'meaning\s+of']
        }

    def analyze_question(self, question: str) -> dict:
        """
        Perform comprehensive analysis of the question to determine its characteristics
        """
        analysis = {
            'platforms': self.identify_platforms(question),
            'question_types': self.identify_question_types(question),
            'complexity': self.assess_complexity(question),
            'components': self.extract_components(question)
        }
        return analysis

    def identify_platforms(self, question: str) -> List[str]:
        """Identify all CDP platforms mentioned in the question"""
        question = question.lower()
        mentioned_platforms = []
        for platform, keywords in self.cdp_keywords.items():
            if any(keyword in question for keyword in keywords):
                mentioned_platforms.append(platform)
        return mentioned_platforms

    def identify_question_types(self, question: str) -> List[str]:
        """Identify the types of question being asked"""
        question = question.lower()
        types = []
        for qtype, patterns in self.question_types.items():
            if any(re.search(pattern, question) for pattern in patterns):
                types.append(qtype)
        return types

    def assess_complexity(self, question: str) -> str:
        """Assess the complexity of the question"""
        # Count the number of distinct question types and platforms
        analysis = self.analyze_question(question)
        num_types = len(analysis['question_types'])
        num_platforms = len(analysis['platforms'])

        if num_types > 1 or num_platforms > 1:
            return 'complex'
        elif len(question.split()) > 20:  # Long questions might need more detailed responses
            return 'moderate'
        else:
            return 'simple'

    def extract_components(self, question: str) -> List[str]:
        """Extract key components from the question"""
        # Split compound questions
        components = []
        sentences = re.split(r'[.?!]\s+', question)
        for sentence in sentences:
            if sentence.strip():
                components.append(sentence.strip())
        return components

    def validate_question(self, question: str) -> Tuple[bool, Optional[str]]:
        """Validate if the question is relevant and well-formed"""
        if not question.strip():
            return False, "Please enter a question."

        if len(question) > 500:
            return False, "Question is too long. Please be more specific."

        # Get question analysis
        analysis = self.analyze_question(question)

        # Check if question is related to CDPs
        if not analysis['platforms'] and not any(keyword in question.lower() for keywords in self.cdp_keywords.values() for keyword in keywords):
            return False, "Please ask a question related to CDP platforms (Segment, mParticle, Lytics, or Zeotap)."

        # Check if question type is supported
        if not analysis['question_types']:
            return False, "Please ask a 'how-to', comparison, or specific implementation question about CDP platforms."

        return True, None