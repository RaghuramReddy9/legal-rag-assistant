import time
from qa_system import load_vectorstore, build_streaming_rag_chain

# === Test Questions (edit as needed) ===
TEST_QUERIES = [
    "What are the supplier’s main obligations under the agreement?",
    "How is data protection handled?",
    "What happens if the customer terminates the contract?",
    "Who owns the intellectual property rights?",
    "Under what law is the agreement governed?",
]

# === Step 1: Load Vectorstore & Chain ===
print(" Loading RAG system...")
vectorstore = load_vectorstore("data/faiss_index")
rag_chain = build_streaming_rag_chain(vectorstore)

# === Step 2: Evaluate ===
total_time = 0
results = []
for q in TEST_QUERIES:
    print(f"\n Question: {q}")
    start = time.time()
    answer = rag_chain.invoke(q)
    duration = time.time() - start
    total_time += duration
    print(f" Answer (first 300 chars):\n{answer[:300]}\n⏱️ Time: {duration:.2f}s")
    results.append({"query": q, "time": duration})

# === Step 3: Summary ===
avg_time = total_time / len(TEST_QUERIES)
print("\n✅ Evaluation Complete")
print(f"Average Response Time: {avg_time:.2f} seconds")
