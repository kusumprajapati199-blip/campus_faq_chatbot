from retrieval_models import KeywordMatcher, TFIDFMatcher, EmbeddingMatcher

DATA_PATH = "data/campus_faqs.json"
query = "I misplaced my college badge, what's the procedure to get a new one?"

print(f"User Query: {query}\n" + "-" * 50)

# 1. Keyword
km = KeywordMatcher(DATA_PATH)
res_km = km.match(query)
print(f"[Keyword Matcher]   Score: {res_km['score']} | Target ID: {res_km['match_id']}")

# 2. TF-IDF
tf = TFIDFMatcher(DATA_PATH)
res_tf = tf.match(query)
print(f"[TF-IDF Matcher]    Score: {res_tf['score']} | Target ID: {res_tf['match_id']}")

# 3. Dense Embedding
emb = EmbeddingMatcher(DATA_PATH)
res_emb = emb.match(query)
print(f"[Embedding Matcher] Score: {res_emb['score']} | Target ID: {res_emb['match_id']}")
if res_emb["faq"]:
    print(f"Matched FAQ Question: {res_emb['faq']['question']}")
    print(f"Retrieved Answer: {res_emb['faq']['answer']}")
