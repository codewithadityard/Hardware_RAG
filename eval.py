import requests
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM

# ==========================================
# 1. CUSTOM LOCAL LLM JUDGE FOR DEEPEVAL
# ==========================================
class LocalOllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        """Routes DeepEval's grading prompts to your local Ollama instance."""
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error communicating with Ollama Judge: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        """DeepEval requires an async method, we just map it to the sync one."""
        return self.generate(prompt)

    def get_model_name(self):
        return f"Local Ollama ({self.model_name})"

# ==========================================
# 2. EVALUATION ROUTINE
# ==========================================
def run_evaluation(retriever, reranker, generator):
    print("Booting RAG Pipeline for Evaluation...")
    
    test_cases_data = [
        {
            "query": "What is the operating voltage of the ESP32?",
            "expected_output": "The ESP32 operates at a voltage range of 2.2V to 3.6V, typically 3.3V."
        },
        {
            "query": "Is Pin 34 an input-only pin?",
            "expected_output": "Yes, GPIO 34 is an input-only pin and does not have internal pull-up or pull-down resistors."
        }
    ]

    # Instantiate our local judge
    local_judge = LocalOllamaJudge(model_name="llama3")

    # Pass the local judge into the metrics so it stops looking for OpenAI
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=local_judge)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=local_judge)
    recall_metric = ContextualRecallMetric(threshold=0.7, model=local_judge)

    for i, data in enumerate(test_cases_data):
        print(f"\n--- Running Test {i+1}: {data['query']} ---")
        
        # 1. Retrieve (Wide Net)
        hybrid_candidates = retriever.retrieve(data['query'], top_k=15)
        
        # 2. Rerank (Precision Filter)
        best_docs = reranker.rerank(data['query'], hybrid_candidates, top_k=3)
        context_list = [doc.page_content for doc in best_docs]

        # 3. Generate Answer
        actual_answer = generator.generate_answer(data['query'], best_docs)
        print(f"\nGenerated Answer: {actual_answer}")

        # 4. Package for DeepEval
        test_case = LLMTestCase(
            input=data["query"],
            actual_output=actual_answer,
            expected_output=data["expected_output"],
            retrieval_context=context_list
        )
        
        # 5. Measure and Print
        print("Scoring metrics with Local Llama 3 Judge... (This may take a minute)")
        
        try:
            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)
            recall_metric.measure(test_case)
            
            print(f" Answer Relevancy Score: {relevancy_metric.score:.2f}")
            print(f"  Faithfulness Score:     {faithfulness_metric.score:.2f}")
            print(f"  Contextual Recall:      {recall_metric.score:.2f}")
            
            if not faithfulness_metric.is_successful():
                print(f" Hallucination Warning: {faithfulness_metric.reason}")
        except Exception as e:
            print(f"\n[Scoring Error] The local judge failed to parse the metrics: {e}")