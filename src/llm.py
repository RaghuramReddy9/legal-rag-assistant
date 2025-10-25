from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-2.0-flash-001"),
    temperature=float(os.getenv("TEMPERATURE", 0.3)),
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

