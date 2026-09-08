import requests
import os
from groq import Groq 
from dotenv import load_dotenv
load_dotenv()

class LocalLLMGenerator:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name

        api_key=os.environ.get("GROQ_API_KEY")

        self.client=Groq(api_key=api_key)

        


    def generate_answer(self,query:str,context_docs):
        if not context_docs:
            return "No relevant data found in the hardware documents "
        
        context_strings = []
        for i, doc in enumerate(context_docs):
            page_num=doc.metadata.get('page',doc.metadata.get('page','Unknown'))
            context_strings.append(f"[Source {i+1} - Page {page_num}]:\n{doc.page_content}")

        context_block = "\n\n---\n\n".join(context_strings)

        prompt = f"""You are a precise electronics engineering assistant.
Answer the user's query using ONLY the hardware specifications provided below.
If the answer is not contained in the context, say "I cannot find this in the datasheet." Do not guess.

CONTEXT:
{context_block}

QUERY: {query}
ANSWER:"""

        print(f"\nSending context to local {self.model_name} model...")

        try:

            response = self.client.chat.completions.create(
                        model=self.model_name, 
                        messages=[ { "role": "user", "content": prompt } ], 
                        temperature=0 )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error:{e}.No answer is generated.")
        
        
            
        