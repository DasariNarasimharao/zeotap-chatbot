import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict, Any
import re
from rank_bm25 import BM25Okapi
from collections import Counter

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
        # Enhanced TF-IDF vectorizer with better parameters
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Include trigrams for better phrase matching
            max_features=10000,
            stop_words='english',
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Apply sublinear scaling to term frequencies
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.documents = {
            'segment': ["This is a sample Segment document."],
            'mparticle': ["This is a sample mParticle document."],
            'lytics': ["This is a sample Lytics document."],
            'zeotap': ["This is a sample Zeotap document."]
        }
        self.document_metadata = {}
        self.bm25 = None
        self._update_vectors()

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with advanced cleaning and normalization"""
        try:
            # Advanced text cleaning
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'\d+', ' NUM ', text)  # Normalize numbers

            # Enhanced tokenization with sentence context
            sentences = sent_tokenize(text)
            all_tokens = []

            for sentence in sentences:
                # Tokenize and tag parts of speech
                tokens = word_tokenize(sentence)

                # Advanced token processing
                processed_tokens = []
                for token in tokens:
                    if (token.isalnum() and 
                        token not in self.stop_words and 
                        len(token) > 1):  # Filter single characters
                        # Lemmatize with context
                        lemma = self.lemmatizer.lemmatize(token)
                        processed_tokens.append(lemma)

                all_tokens.extend(processed_tokens)

            return ' '.join(all_tokens)
        except Exception as e:
            print(f"Error in preprocess_text: {e}")
            return text

    def calculate_term_importance(self, term: str, doc_tokens: List[str]) -> float:
        """Calculate term importance based on position and frequency"""
        if not doc_tokens:
            return 0.0

        # Get term frequency
        term_freq = doc_tokens.count(term)

        # Get positions of term in document
        positions = [i for i, t in enumerate(doc_tokens) if t == term]

        # Calculate position-based importance (terms appearing earlier are more important)
        position_importance = sum(1 / (pos + 1) for pos in positions) if positions else 0

        return term_freq * (1 + position_importance)

    def add_documents(self, platform: str, documents: List[str], metadata: Dict[str, Any] = None):
        """Add documents with metadata and update search indices"""
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        self.documents[platform.lower()] = processed_docs

        if metadata:
            self.document_metadata[platform.lower()] = metadata

        # Update search indices
        self._update_vectors()
        self._update_bm25_index()

    def _update_vectors(self):
        """Update TF-IDF vectors and BM25 index"""
        try:
            all_docs = []
            for platform_docs in self.documents.values():
                all_docs.extend(platform_docs)

            if all_docs:
                # Update TF-IDF vectors
                self.document_vectors = self.vectorizer.fit_transform(all_docs)

                # Create tokenized documents for BM25
                tokenized_docs = [doc.split() for doc in all_docs]
                self.bm25 = BM25Okapi(tokenized_docs)
        except Exception as e:
            print(f"Error in _update_vectors: {e}")
            self.document_vectors = None
            self.bm25 = None

    def get_combined_similarity(self, query: str, doc: str, tfidf_sim: float, bm25_score: float) -> float:
        """Combine different similarity metrics with weighted scoring"""
        # Preprocess query and document
        processed_query = self.preprocess_text(query)
        processed_doc = self.preprocess_text(doc)

        # Calculate term overlap
        query_terms = set(processed_query.split())
        doc_terms = set(processed_doc.split())
        term_overlap = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0

        # Combine scores with weights
        weights = {
            'tfidf': 0.4,
            'bm25': 0.4,
            'term_overlap': 0.2
        }

        combined_score = (
            weights['tfidf'] * tfidf_sim +
            weights['bm25'] * (bm25_score / 10) +  # Normalize BM25 score
            weights['term_overlap'] * term_overlap
        )

        return combined_score

    def find_relevant_documents(self, query: str, platform: str = None, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find relevant documents using multiple ranking methods"""
        try:
            processed_query = self.preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])

            if platform:
                # Search in specific platform
                platform_docs = self.documents.get(platform.lower(), [])
                if not platform_docs:
                    return [("No documents available for this platform.", 0.0)]

                # Get TF-IDF similarities
                platform_vectors = self.vectorizer.transform(platform_docs)
                tfidf_similarities = cosine_similarity(query_vector, platform_vectors)[0]

                # Get BM25 scores
                query_tokens = processed_query.split()
                bm25_scores = self.bm25.get_scores(query_tokens)

                # Combine scores
                combined_scores = [
                    self.get_combined_similarity(query, doc, tfidf_sim, bm25_score)
                    for doc, tfidf_sim, bm25_score in zip(platform_docs, tfidf_similarities, bm25_scores)
                ]

                # Get top results
                top_indices = np.argsort(combined_scores)[-top_k:][::-1]
                return [(platform_docs[i], combined_scores[i]) for i in top_indices]
            else:
                # Search across all documents
                if not self.document_vectors:
                    return [("No documents available.", 0.0)]

                all_docs = [doc for docs in self.documents.values() for doc in docs]

                # Get combined scores for all documents
                combined_scores = []
                tfidf_similarities = cosine_similarity(query_vector, self.document_vectors)[0]
                query_tokens = processed_query.split()
                bm25_scores = self.bm25.get_scores(query_tokens)

                for doc, tfidf_sim, bm25_score in zip(all_docs, tfidf_similarities, bm25_scores):
                    score = self.get_combined_similarity(query, doc, tfidf_sim, bm25_score)
                    combined_scores.append(score)

                top_indices = np.argsort(combined_scores)[-top_k:][::-1]
                return [(all_docs[i], combined_scores[i]) for i in top_indices]

        except Exception as e:
            print(f"Error in find_relevant_documents: {e}")
            return [(f"An error occurred while processing your query: {str(e)}", 0.0)]

    def generate_response(self, query: str, question_analysis: dict) -> Tuple[str, float]:
        """Generate a comprehensive response using enhanced matching"""
        try:
            platforms = question_analysis['platforms']
            complexity = question_analysis['complexity']
            question_types = question_analysis['question_types']

            # Handle comparison questions
            if 'comparison' in question_types and len(platforms) > 1:
                responses = []
                max_similarity = 0.0

                for platform in platforms:
                    docs = self.find_relevant_documents(query, platform, top_k=2)
                    if docs[0][1] > 0.3:
                        responses.append(f"{platform.title()}: {docs[0][0]}")
                        max_similarity = max(max_similarity, docs[0][1])

                if responses:
                    return "Comparison:\n" + "\n\n".join(responses), max_similarity

            # Handle complex questions
            if complexity == 'complex':
                all_responses = []
                max_similarity = 0.0

                # Process each component of the complex question
                for component in question_analysis['components']:
                    docs = self.find_relevant_documents(component, platforms[0] if platforms else None)
                    if docs[0][1] > 0.3:
                        all_responses.append(docs[0][0])
                        max_similarity = max(max_similarity, docs[0][1])

                if all_responses:
                    formatted_response = "\n\nStep-by-step answer:\n" + "\n".join(
                        f"{i+1}. {resp}" for i, resp in enumerate(all_responses)
                    )
                    return formatted_response, max_similarity

            # Default to best matching response
            docs = self.find_relevant_documents(query, platforms[0] if platforms else None)
            return docs[0][0], docs[0][1]

        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"An error occurred while generating the response: {str(e)}", 0.0

    def _update_bm25_index(self):
        """Update the BM25 index."""
        try:
            all_docs = []
            for platform_docs in self.documents.values():
                all_docs.extend(platform_docs)
            if all_docs:
                self.tokenized_docs = [doc.split() for doc in all_docs]
                self.bm25 = BM25Okapi(self.tokenized_docs)

        except Exception as e:
            print(f"Error updating BM25 index: {e}")
            self.bm25 = None