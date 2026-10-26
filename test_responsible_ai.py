from responsible_ai import ResponsibleChatbotAgent

agent = ResponsibleChatbotAgent(data_path="data/campus_faqs.json", confidence_threshold=0.45)

print("\n--- TEST 1: Valid In-Domain Query ---")
res1 = agent.process_query("Can you tell me how to get my misplaced ID replaced?", selected_method="Sentence-Transformers")
print(f"Status: {res1['status']}")
print(f"Matched Question: {res1['matched_question']}")
print(f"Answer: {res1['answer']}")
print(f"Explanation: {res1['explainability']}")

print("\n--- TEST 2: Out-of-Domain Query (Triggering Safety Fallback) ---")
res2 = agent.process_query("What is the recipe for chocolate cake?", selected_method="Sentence-Transformers")
print(f"Status: {res2['status']}")
print(f"Answer: {res2['answer']}")
print(f"Explanation: {res2['explainability']}")
