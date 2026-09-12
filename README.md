ESP32 Hardware RAG Intelligence 🧠⚡

A production-grade Retrieval-Augmented Generation (RAG) pipeline engineered specifically for embedded systems. This project parses, searches, and reasons over the 700-page ESP32 Technical Reference Manual, addressing the unique challenges of hardware documentation such as dense pinout tables, exact register hex addresses, and complex multi-hop hardware constraints.

🏗️ Core Architecture & Features
1.Custom Table Linearization: Utilizes PyMuPDF to programmatically flatten complex multi-column memory and pinout tables, preventing context severing during the PDF ingestion phase.

2.Query Transformation: Intercepts and rewrites raw user prompts into optimized target vectors to improve retrieval accuracy for highly technical hardware queries.

3.Hybrid Search with RRF: Merges dense semantic vector search (FAISS) with sparse lexical keyword matching (BM25) using Reciprocal Rank Fusion (RRF) to capture both conceptual intent and exact register names.

4.Cross-Encoder Dynamic Thresholding: Aggressively filters out low-confidence noise chunks by calculating a dynamic drop-off delta before context is passed to the generation layer.

5.Lost-in-the-Middle Reordering: Physically shuffles the retrieved context array to place the most critical chunks at the very top and bottom of the prompt, bypassing standard Transformer attention amnesia.

6.White-Box Observability: Features a custom RAG Debugger providing step-by-step terminal telemetry, displaying logit scores, threshold cutoffs, and context layouts.

7.Redis Caching: Bypasses LLM generation for repeated queries, dropping pipeline latency from ~4.0s to 0.01s.

🛠️ Tech Stack
Language: Python

Generation: Groq API (openai/gpt-oss-120b)

Retrieval: FAISS, BM25, Cross-Encoder (HuggingFace)

PDF Processing: PyMuPDF (fitz)

Caching: Redis

Evaluation: DeepEval (LLM-as-a-Judge)

🚀 Quick Start
Prerequisites
Python 3.10+

Redis Server running locally on port 6379

Groq API Key

Installation
Bash
git clone https://github.com/yourusername/ESP32-Hardware-RAG.git
cd ESP32-Hardware-RAG
pip install -r requirements.txt
Environment Variables
Create a .env file in the root directory:

Plaintext
GROQ_API_KEY=your_groq_api_key_here
Execution
Bash
python main.py
Upon execution, select 1 for the Interactive Hardware Assistant Terminal or 2 for the Performance Benchmarking Suite.

🧪 Automated Benchmarking (DeepEval)
In embedded systems, hallucinating a pin state or register offset can cause catastrophic hardware failure. This pipeline relies on mathematical regression testing rather than standard prompting.

The system includes a 15-question Golden Dataset targeting:

a.Factual Table & Spec Lookups

b.Hex Addresses & Lexical Matching

c.Safe Rejections / Out-of-Scope Checks

d.Executing the Benchmarking Suite triggers DeepEval using a 70B parameter model as an LLM-as-a-Judge to evaluate the pipeline mathematically on Answer Relevancy, Faithfulness, and Contextual Recall.

📊 Observability & Telemetry
The interactive terminal includes a built-in RAG Debugger. 
Typing y when prompted for the diagnostic trace outputs the exact pipeline execution path, including:
the query routing strategy, the cross-encoder threshold math, and the specific arrangement of the Lost-in-the-Middle context window.
