from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)


def filter_chunks_by_keyword(chunks: List[str], keyword: str) -> List[str]:
    """
    Returns chunks that contain the keyword (case-insensitive).
    """
    return [chunk for chunk in chunks if keyword.lower() in chunk.lower()]


def score_chunks(chunks: List[str], keyword: str) -> Dict[str, Dict[str, int]]:
    """
    Returns dict of chunks and how many times the keyword appears in each.
    """
    return {
        f"chunk_{i}": {
            "text": chunk,
            "score": chunk.lower().count(keyword.lower())
        }
        for i, chunk in enumerate(chunks)
    }


def top_chunks_by_score(scored_dict: Dict[str, Dict[str, int]], top_n: int = 3) -> List[Dict[str, Dict[str, int]]]:
    """
    Returns top N chunks with the highest keyword match score.
    """
    return sorted(
        scored_dict.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )[:top_n]
