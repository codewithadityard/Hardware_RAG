import requests
import os
from groq import Groq 
from dotenv import load_dotenv

load_dotenv()

class QueryTransformer:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        api_key=os.environ.get("GROQ_API_KEY")

        if not api_key: 
            raise ValueError( "GROQ_API_KEY not found. Please add it to your .env file." )

        self.client=Groq(api_key=api_key)

    def rewrite_query(self, raw_query: str) -> str:
        prompt = f"""You are an expert electronics engineer. 
Your task is to rewrite the following user query to be optimized for a vector database search over a microcontroller datasheet.

RULES:
1. Fix typos and expand casual abbreviations (e.g., 'esp' to 'ESP32').
2. If the user asks a specific technical question, add relevant hardware keywords (e.g., 'pins' -> 'GPIO').
3. CRITICAL: If the user asks a general, broad, or introductory question (e.g., "what is...", "tell me about..."), DO NOT add unprompted technical requirements like voltage or architecture. Keep it broad!

Output ONLY the rewritten query. Do not include introductory text, quotes, or explanations.

Raw Query: {raw_query}
Rewritten Query:"""

        
        try:
           response = self.client.chat.completions.create(
              model=self.model_name, 
              messages=[ { "role": "user", "content": prompt } ], 
              temperature=0 )
           optimized_query = response.choices[0].message.content.strip( "\"' " )
           return optimized_query
            
        except Exception as e:
            print(f"[Warning] Query transformation failed: {e}. Falling back to raw query.")
            return raw_query