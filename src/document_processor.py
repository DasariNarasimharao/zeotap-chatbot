import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        self.vectorizer = TfidfVectorizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.documents = {
            'segment': ["This is a sample Segment document."],
            'mparticle': ["This is a sample mParticle document."],
            'lytics': ["This is a sample Lytics document."],
            'zeotap': ["This is a sample Zeotap document."]
        }
        self._update_vectors()

    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing"""
        try:
            # Basic cleaning
            text = text.lower().strip()

            # Tokenization with error handling
            try:
                tokens = word_tokenize(text)
            except Exception:
                tokens = text.split()

            # Remove stopwords and non-alphanumeric tokens
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token.isalnum() and token not in self.stop_words
            ]

            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in preprocess_text: {e}")
            return text

    def add_documents(self, platform, documents):
        """Add documents for a specific CDP platform"""
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        self.documents[platform.lower()] = processed_docs
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

    def find_relevant_document(self, query, platform=None):
        """Find the most relevant document for a given query"""
        try:
            processed_query = self.preprocess_text(query)

            if platform:
                # Search only in specific platform documents
                platform_docs = self.documents[platform.lower()]
                if not platform_docs:
                    return "No documents available for this platform.", 0.0

                platform_vectors = self.vectorizer.transform(platform_docs)
                query_vector = self.vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, platform_vectors)[0]
                best_idx = np.argmax(similarities)
                return platform_docs[best_idx], similarities[best_idx]
            else:
                # Search across all documents
                if not self.document_vectors:
                    return "No documents available.", 0.0

                query_vector = self.vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.document_vectors)[0]
                best_idx = np.argmax(similarities)
                all_docs = [doc for docs in self.documents.values() for doc in docs]
                return all_docs[best_idx], similarities[best_idx]

        except Exception as e:
            print(f"Error in find_relevant_document: {e}")
            return f"An error occurred while processing your query: {str(e)}", 0.0