import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------
# 1. Base Class & Data Loader
# -------------------------------------------------------------
class FAQRetriever:
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)
        self.questions = [item["question"] for item in self.faqs]
        self.ids = [item["id"] for item in self.faqs]

# -------------------------------------------------------------
# 2. Method 1: Keyword Matching (Jaccard Overlap Baseline)
# -------------------------------------------------------------
class KeywordMatcher(FAQRetriever):
    def _tokenize(self, text: str) -> set:
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        return set(clean_text.split())

    def match(self, query: str):
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {"match_id": None, "score": 0.0, "faq": None}

        best_score = 0.0
        best_idx = -1

        for idx, question in enumerate(self.questions):
            q_tokens = self._tokenize(question)
            union = query_tokens.union(q_tokens)
            if not union:
                continue
            jaccard_score = len(query_tokens.intersection(q_tokens)) / len(union)
            
            if jaccard_score > best_score:
                best_score = jaccard_score
                best_idx = idx

        if best_idx != -1 and best_score > 0.0:
            return {
                "match_id": self.ids[best_idx],
                "score": round(float(best_score), 4),
                "faq": self.faqs[best_idx]
            }
        return {"match_id": None, "score": 0.0, "faq": None}

# -------------------------------------------------------------
# 3. Method 2: TF-IDF + Cosine Similarity
# -------------------------------------------------------------
class TFIDFMatcher(FAQRetriever):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def match(self, query: str):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score > 0.0:
            return {
                "match_id": self.ids[best_idx],
                "score": round(best_score, 4),
                "faq": self.faqs[best_idx]
            }
        return {"match_id": None, "score": 0.0, "faq": None}

# -------------------------------------------------------------
# 4. Method 3: Sentence-Transformers (Dense Semantic Embeddings)
# -------------------------------------------------------------
class EmbeddingMatcher(FAQRetriever):
    def __init__(self, data_path: str, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(data_path)
        # Lightweight, high-accuracy Hugging Face model
        self.model = SentenceTransformer(model_name)
        # Compute & cache embeddings for FAQ questions
        self.question_embeddings = self.model.encode(self.questions, normalize_embeddings=True)

    def match(self, query: str):
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        # Dot product equals cosine similarity when vectors are normalized
        similarities = np.dot(query_embedding, self.question_embeddings.T)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        return {
            "match_id": self.ids[best_idx],
            "score": round(best_score, 4),
            "faq": self.faqs[best_idx]
        }
