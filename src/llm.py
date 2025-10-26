import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Return a configured Gemini 2.0 Flash model."""
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_NAME", "gemini-2.0-flash-001"),
        temperature=float(os.getenv("TEMPERATURE", 0.3)),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return llm
