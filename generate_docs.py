from fpdf import FPDF
import os

class ProjectDocumentationPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("helvetica", "", 24)

    def header(self):
        if self.page_no() > 1:  # Skip header on first page
            self.set_font("helvetica", "", 10)
            self.cell(0, 10, f"CDP Support Chatbot Documentation - Page {self.page_no()}", align="C")
            self.ln(20)

    def chapter_title(self, title):
        self.set_font("helvetica", "B", 20)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(10)

    def chapter_body(self, body):
        self.set_font("helvetica", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def section_title(self, title):
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def bullet_point(self, text):
        self.set_font("helvetica", "", 12)
        bullet_width = 10
        available_width = self.epw - bullet_width  # Calculate available width
        self.cell(bullet_width, 10, "-", new_x="RIGHT")
        self.multi_cell(available_width, 10, text)

def generate_documentation():
    pdf = ProjectDocumentationPDF()

    # Title Page
    pdf.cell(0, 60, "CDP Support Chatbot", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "I", 16)
    pdf.cell(0, 20, "Technical Documentation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Generated on: {os.popen('date').read().strip()}", align="C")

    # Table of Contents
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    sections = [
        "1. Project Overview",
        "2. Core Features",
        "3. Technical Architecture",
        "4. Implementation Details",
        "5. User Interface",
        "6. Advanced Features",
        "7. Future Enhancements"
    ]
    for section in sections:
        pdf.cell(0, 10, section, new_x="LMARGIN", new_y="NEXT")

    # Project Overview
    pdf.add_page()
    pdf.chapter_title("1. Project Overview")
    pdf.chapter_body("""
The CDP Support Chatbot is a sophisticated Streamlit-based application designed to provide comprehensive and nuanced answers to complex how-to questions about Customer Data Platforms (CDPs). The system leverages advanced similarity matching and multi-CDP documentation integration to deliver accurate and relevant responses.

Key Supported Platforms:
- Segment
- mParticle
- Lytics
- Zeotap
    """)

    # Core Features
    pdf.add_page()
    pdf.chapter_title("2. Core Features")

    features = [
        "Advanced Question Processing with multi-part query support",
        "Multi-Platform CDP Support with detailed documentation integration",
        "Enhanced Similarity Matching using TF-IDF and BM25",
        "Platform Comparison Analysis with key difference detection",
        "Interactive User Interface with confidence indicators"
    ]

    for feature in features:
        pdf.bullet_point(feature)

    # Technical Architecture
    pdf.add_page()
    pdf.chapter_title("3. Technical Architecture")

    pdf.section_title("Core Components")
    components = {
        "Question Handler": "Advanced query analysis and preprocessing",
        "Document Processor": "Enhanced similarity matching and response generation",
        "Comparison Engine": "Cross-platform feature comparison and analysis",
        "UI Layer": "Streamlit-based interactive interface with real-time feedback"
    }

    for component, description in components.items():
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, component, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 12)
        pdf.multi_cell(0, 10, description)
        pdf.ln(5)

    # Implementation Details
    pdf.add_page()
    pdf.chapter_title("4. Implementation Details")

    implementations = [
        ("Question Analysis", """
- Advanced tokenization and preprocessing
- Intent classification
- Complexity assessment
- Multi-part question breakdown"""),
        ("Similarity Matching", """
- TF-IDF vectorization
- BM25 ranking
- Hybrid scoring system
- Context-aware response selection"""),
        ("Platform Comparison", """
- Cross-platform feature analysis
- Key difference extraction
- Term-based comparison
- Structured comparison output""")
    ]

    for title, details in implementations:
        pdf.section_title(title)
        pdf.multi_cell(0, 10, details)
        pdf.ln(10)

    # User Interface
    pdf.add_page()
    pdf.chapter_title("5. User Interface")

    ui_features = [
        "Clean and intuitive chat interface",
        "Platform selection dropdown",
        "Question type hints",
        "Confidence level indicators",
        "Example questions section",
        "Response analysis display"
    ]

    for feature in ui_features:
        pdf.bullet_point(feature)

    # Advanced Features
    pdf.add_page()
    pdf.chapter_title("6. Advanced Features")

    advanced_features = [
        "Cross-Platform Comparison Engine",
        "Enhanced Similarity Matching with BM25",
        "Multi-part Question Processing",
        "Context-Aware Response Generation",
        "Confidence Scoring System"
    ]

    for feature in advanced_features:
        pdf.bullet_point(feature)

    # Future Enhancements
    pdf.add_page()
    pdf.chapter_title("7. Future Enhancements")

    future_features = [
        "Advanced visual comparison guides",
        "Enhanced analytics dashboard",
        "Custom training data integration",
        "API endpoint integration",
        "Real-time documentation updates"
    ]

    for feature in future_features:
        pdf.bullet_point(feature)

    # Save the PDF
    pdf.output("CDP_Support_Chatbot_Documentation.pdf")

if __name__ == "__main__":
    generate_documentation()