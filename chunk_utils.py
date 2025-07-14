from typing import List, Dict

def filter_chunks_by_keyword(chunks: List[str], key:str) -> list[str]:
    """
    Filters a list of text chunks for those containing the keyword (case-insensitive).
    """
    result = []
    for chunk in chunks:
        if key.lower() in chunk.lower():
            result.append(chunk)
    return result


def score_chunks(chunks: List[str], key: str) -> Dict[str, Dict[str, int]]:
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


def top_chunks_by_score(scored_dict, top_n=3):
    sorted_chunks = sorted(
        scored_dict.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )
    return sorted_chunks[:top_n]



chunks = [
    "Refunds are available within 30 days of purchase.",
    "This document describes payment policies and terms.",
    "No refund will be issued after 30 days.",
]

filtered = filter_chunks_by_keyword(chunks, "refund")
print(f"Filtered Chunks:\n", filtered)

scored = score_chunks(chunks, "refund")
print(f"Scored Chunks:\n", scored)




