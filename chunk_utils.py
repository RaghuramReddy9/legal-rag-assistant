def filter_chunks_by_keyword(chunks, key):
    """
    Filters a list of text chunks for those containing the keyword (case-insensitive).
    """
    result = []
    for chunk in chunks:
        if key.lower() in chunk.lower():
            result.append(chunk)
    return result


def score_chunks(chunks, key):
    """
    Scores each chunk based on how many times the keyword appears.
    Returns a dictionary with chunk text and keyword frequency.
    """
    score = {}
    for i, chunk in enumerate(chunks):
        count = chunk.lower().count(key.lower())
        score[f"chunk_{i}"] = {
            "text": chunk,
            "score": count
        }
    return score


chunks = [
    "Refunds are available within 30 days of purchase.",
    "This document describes payment policies and terms.",
    "No refund will be issued after 30 days.",
]

filtered = filter_chunks_by_keyword(chunks, "refund")
print(f"Filtered Chunks:\n", filtered)

scored = score_chunks(chunks, "refund")
print(f"Scored Chunks:\n", scored)




