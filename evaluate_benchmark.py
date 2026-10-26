import json
import time
import pandas as pd
from retrieval_models import KeywordMatcher, TFIDFMatcher, EmbeddingMatcher

DATA_PATH = "data/campus_faqs.json"
TEST_PATH = "data/test_queries.json"

def run_evaluation():
    # 1. Load Paraphrased Benchmark Test Set
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 2. Instantiate all three models
    models = {
        "Keyword (Jaccard)": KeywordMatcher(DATA_PATH),
        "TF-IDF + Cosine": TFIDFMatcher(DATA_PATH),
        "Sentence-Transformers": EmbeddingMatcher(DATA_PATH)
    }

    results = []
    detailed_logs = []

    # 3. Benchmark Each Model
    for model_name, model in models.items():
        correct_hits = 0
        total_queries = len(test_data)
        latencies = []

        for item in test_data:
            query = item["test_query"]
            target_id = item["target_faq_id"]

            # Measure inference time per query
            start_time = time.perf_counter()
            match_result = model.match(query)
            latency = (time.perf_counter() - start_time) * 1000  # in ms
            latencies.append(latency)

            is_correct = (match_result["match_id"] == target_id)
            if is_correct:
                correct_hits += 1

            detailed_logs.append({
                "Model": model_name,
                "Query": query,
                "Target_ID": target_id,
                "Predicted_ID": match_result["match_id"],
                "Confidence_Score": match_result["score"],
                "Is_Correct": is_correct
            })

        accuracy = (correct_hits / total_queries) * 100
        avg_latency = sum(latencies) / len(latencies)

        results.append({
            "Retrieval Method": model_name,
            "Accuracy (Hit@1 %)": round(accuracy, 2),
            "Avg Latency (ms)": round(avg_latency, 3),
            "Correct": f"{correct_hits}/{total_queries}"
        })

    # 4. Display Results Table
    df_summary = pd.DataFrame(results)
    print("\n" + "="*60)
    print("      CSE276: COMPARATIVE PERFORMANCE EVALUATION")
    print("="*60)
    print(df_summary.to_string(index=False))
    print("="*60 + "\n")

    # 5. Export Logs for Failure and Error Analysis
    df_logs = pd.DataFrame(detailed_logs)
    df_logs.to_csv("data/evaluation_detailed_log.csv", index=False)
    print("Saved detailed failure logs to data/evaluation_detailed_log.csv\n")

if __name__ == "__main__":
    run_evaluation()
