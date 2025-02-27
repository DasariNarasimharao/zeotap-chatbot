import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class DocumentProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.documents = {
            'segment': [],
            'mparticle': [],
            'lytics': [],
            'zeotap': []
        }
        self.document_vectors = None
        
    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing"""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        return ' '.join(tokens)
    
    def add_documents(self, platform, documents):
        """Add documents for a specific CDP platform"""
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        self.documents[platform.lower()] = processed_docs
        self._update_vectors()
    
    def _update_vectors(self):
        """Update TF-IDF vectors for all documents"""
        all_docs = []
        for platform_docs in self.documents.values():
            all_docs.extend(platform_docs)
        if all_docs:
            self.document_vectors = self.vectorizer.fit_transform(all_docs)
    
    def find_relevant_document(self, query, platform=None):
        """Find the most relevant document for a given query"""
        processed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        
        if platform:
            # Search only in specific platform documents
            platform_docs = self.documents[platform.lower()]
            platform_vectors = self.vectorizer.transform(platform_docs)
            similarities = cosine_similarity(query_vector, platform_vectors)[0]
            best_idx = np.argmax(similarities)
            return platform_docs[best_idx], similarities[best_idx]
        else:
            # Search across all documents
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            best_idx = np.argmax(similarities)
            return list(self.documents.values())[best_idx], similarities[best_idx]
