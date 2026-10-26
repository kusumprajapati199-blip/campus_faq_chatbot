from retrieval_models import EmbeddingMatcher, TFIDFMatcher, KeywordMatcher

class ResponsibleChatbotAgent:
    def __init__(self, data_path: str, confidence_threshold: float = 0.50):
        self.data_path = data_path
        self.confidence_threshold = confidence_threshold
        
        # Load matchers
        self.models = {
            "Sentence-Transformers": EmbeddingMatcher(data_path),
            "TF-IDF": TFIDFMatcher(data_path),
            "Keyword (Jaccard)": KeywordMatcher(data_path)
        }

    def process_query(self, query: str, selected_method: str = "Sentence-Transformers"):
        """
        Executes query retrieval with explainability traces and Responsible AI guards.
        """
        matcher = self.models.get(selected_method)
        if not matcher:
            raise ValueError(f"Unknown retrieval method: {selected_method}")

        # 1. Retrieve Candidate
        result = matcher.match(query)
        score = result["score"]
        faq = result["faq"]

        # 2. Responsible AI Guardrail: Confidence Thresholding (Preventing Hallucination)
        if faq is None or score < self.confidence_threshold:
            return {
                "status": "FALLBACK",
                "answer": (
                    "I could not locate an official university policy matching your inquiry with high confidence. "
                    "To prevent misinformation, please consult the relevant administrative window directly."
                ),
                "matched_question": None,
                "confidence_score": score,
                "method_used": selected_method,
                "explainability": {
                    "reasoning": f"Confidence score ({score:.4f}) fell below the safety threshold of {self.confidence_threshold}.",
                    "action_taken": "Safe fallback triggered to prevent false information."
                }
            }

        # 3. Explainability Trace
        explanation = self._generate_explanation(query, faq["question"], score, selected_method)

        return {
            "status": "SUCCESS",
            "answer": faq["answer"],
            "category": faq["category"],
            "matched_question": faq["question"],
            "confidence_score": score,
            "method_used": selected_method,
            "explainability": explanation
        }

    def _generate_explanation(self, query: str, matched_q: str, score: float, method: str):
        if method == "Keyword (Jaccard)":
            q_set = set(query.lower().split())
            m_set = set(matched_q.lower().split())
            common_words = list(q_set.intersection(m_set))
            return {
                "type": "Token Overlap",
                "shared_keywords": common_words,
                "overlap_ratio": score,
                "reasoning": f"Retrieved because of {len(common_words)} overlapping lexical tokens."
            }
        elif method == "TF-IDF":
            return {
                "type": "Term Frequency-Inverse Document Frequency",
                "cosine_similarity": score,
                "reasoning": "Matched based on shared high-weight domain keywords within the corpus vector space."
            }
        else:
            return {
                "type": "Dense Semantic Embedding Cosine Similarity",
                "vector_similarity": score,
                "reasoning": "Matched using contextual semantic vectors. The model recognizes intent similarity even if specific keywords differ."
            }
