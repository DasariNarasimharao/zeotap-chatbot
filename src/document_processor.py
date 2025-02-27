import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict, Any
import re

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt', download_dir='/home/runner/nltk_data')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

class DocumentProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Include bigrams
            max_features=5000,
            stop_words='english'
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.documents = {
            'segment': ["This is a sample Segment document."],
            'mparticle': ["This is a sample mParticle document."],
            'lytics': ["This is a sample Lytics document."],
            'zeotap': ["This is a sample Zeotap document."]
        }
        self.document_metadata = {}  # Store metadata about documents
        self._update_vectors()

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        try:
            # Basic cleaning
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation

            # Tokenization with error handling
            try:
                sentences = sent_tokenize(text)
                tokens = []
                for sentence in sentences:
                    tokens.extend(word_tokenize(sentence))
            except Exception:
                tokens = text.split()

            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token.isalnum() and token not in self.stop_words
            ]

            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in preprocess_text: {e}")
            return text

    def add_documents(self, platform: str, documents: List[str], metadata: Dict[str, Any] = None):
        """Add documents with metadata for a specific CDP platform"""
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        self.documents[platform.lower()] = processed_docs
        if metadata:
            self.document_metadata[platform.lower()] = metadata
        self._update_vectors()

    def _update_vectors(self):
        """Update TF-IDF vectors for all documents"""
        try:
            all_docs = []
            for platform_docs in self.documents.values():
                all_docs.extend(platform_docs)
            if all_docs:
                self.document_vectors = self.vectorizer.fit_transform(all_docs)
        except Exception as e:
            print(f"Error in _update_vectors: {e}")
            self.document_vectors = None

    def find_relevant_documents(self, query: str, platform: str = None, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find multiple relevant documents for a complex query"""
        try:
            processed_query = self.preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])

            if platform:
                # Search only in specific platform documents
                platform_docs = self.documents.get(platform.lower(), [])
                if not platform_docs:
                    return [("No documents available for this platform.", 0.0)]

                platform_vectors = self.vectorizer.transform(platform_docs)
                similarities = cosine_similarity(query_vector, platform_vectors)[0]

                # Get top-k results
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                return [(platform_docs[i], similarities[i]) for i in top_indices]
            else:
                # Search across all documents
                if not self.document_vectors:
                    return [("No documents available.", 0.0)]

                similarities = cosine_similarity(query_vector, self.document_vectors)[0]
                top_indices = np.argsort(similarities)[-top_k:][::-1]

                all_docs = [doc for docs in self.documents.values() for doc in docs]
                return [(all_docs[i], similarities[i]) for i in top_indices]

        except Exception as e:
            print(f"Error in find_relevant_documents: {e}")
            return [(f"An error occurred while processing your query: {str(e)}", 0.0)]

    def generate_response(self, query: str, question_analysis: dict) -> Tuple[str, float]:
        """Generate a comprehensive response based on question analysis"""
        try:
            platforms = question_analysis['platforms']
            complexity = question_analysis['complexity']
            question_types = question_analysis['question_types']

            # Handle different types of questions
            if 'comparison' in question_types and len(platforms) > 1:
                # Generate comparison response
                responses = []
                for platform in platforms:
                    docs = self.find_relevant_documents(query, platform, top_k=2)
                    if docs[0][1] > 0.3:  # If similarity is above threshold
                        responses.append(f"{platform.title()}: {docs[0][0]}")

                if responses:
                    combined_response = "Comparison:\n" + "\n\n".join(responses)
                    return combined_response, max(doc[1] for p in platforms for doc in self.find_relevant_documents(query, p))

            # Handle complex questions
            if complexity == 'complex':
                all_responses = []
                max_similarity = 0.0

                for component in question_analysis['components']:
                    docs = self.find_relevant_documents(component, platforms[0] if platforms else None)
                    if docs[0][1] > 0.3:
                        all_responses.append(docs[0][0])
                        max_similarity = max(max_similarity, docs[0][1])

                if all_responses:
                    return "\n\nStep-by-step answer:\n" + "\n".join(f"{i+1}. {resp}" for i, resp in enumerate(all_responses)), max_similarity

            # Default to single best response
            docs = self.find_relevant_documents(query, platforms[0] if platforms else None)
            return docs[0][0], docs[0][1]

        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"An error occurred while generating the response: {str(e)}", 0.0