import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm(provider="gemini"):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    elif provider == "openai":
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    else:
        raise ValueError("Unsupported provider")