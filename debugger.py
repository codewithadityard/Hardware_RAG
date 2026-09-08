# src/debugger.py
import time
from typing import List, Dict, Any

class RAGDebugger:
    def __init__(self):
        self.trace: Dict[str, Any] = {}

    def start_trace(self, user_query: str):
        """Initializes a new diagnostic session for a query."""
        self.trace = {
            "user_query": user_query,
            "transformed_query": None,
            "routing_decision": None,
            "retrieval": {
                "dense_candidates": [],
                "sparse_candidates": [],
                "hybrid_merged": []
            },
            "reranking": {
                "scored_documents": [],
                "top_score": None,
                "cutoff_threshold": None,
                "passed_documents": [],
                "dropped_documents": []
            },
            "context_window_layout": [],
            "generation": {
                "final_answer": None,
                "latency_breakdown": {}
            }
        }

    def log_routing(self, transformed_query: str, was_routed: bool):
        self.trace["transformed_query"] = transformed_query
        self.trace["routing_decision"] = "Heavy Transformer" if was_routed else "Semantic Fast-Path (Bypassed)"

    def log_reranking(self, scored_docs_with_metadata: List[Dict], threshold_cutoff: float, top_k_docs: List[Any], reordered_docs: List[Any]):
        top_score = scored_docs_with_metadata[0]["score"] if scored_docs_with_metadata else 0.0
        self.trace["reranking"]["top_score"] = top_score
        self.trace["reranking"]["cutoff_threshold"] = top_score - threshold_cutoff
        
        passed = []
        dropped = []
        for item in scored_docs_with_metadata:
            if item["score"] >= (top_score - threshold_cutoff):
                passed.append(item)
            else:
                dropped.append(item)
                
        self.trace["reranking"]["passed_documents"] = passed[:len(top_k_docs)]
        self.trace["reranking"]["dropped_documents"] = dropped
        self.trace["context_window_layout"] = [
            {"position": idx + 1, "page": getattr(doc, "metadata", {}).get("page", "N/A"), "snippet": doc.page_content[:90] + "..."}
            for idx, doc in enumerate(reordered_docs)
        ]

    def print_diagnostic_report(self):
        """Prints an interactive visual inspection trace to the terminal."""
        t = self.trace
        print("\n" + "=" * 70)
        print("🔍 RAG PIPELINE DIAGNOSTIC TRACE REPORT")
        print("=" * 70)
        
        # 1. Query & Routing Stage
        print(f"\n[1. Query Transformation & Routing]")
        print(f" • Input Query      : \"{t['user_query']}\"")
        print(f" • Routing Strategy : {t['routing_decision']}")
        print(f" • Optimized Target : \"{t['transformed_query']}\"")

        # 2. Reranking & Dynamic Cutoff
        rerank = t["reranking"]
        print(f"\n[2. Cross-Encoder Scoring & Threshold Filter]")
        print(f" • Top Logit Score  : {rerank['top_score']:.4f}")
        print(f" • Threshold Cutoff : {rerank['cutoff_threshold']:.4f} (Dropoff Delta: {rerank['top_score'] - rerank['cutoff_threshold']:.1f})")
        print(f" • Chunks Passed    : {len(rerank['passed_documents'])}")
        print(f" • Chunks Filtered  : {len(rerank['dropped_documents'])} (Dropped as low-confidence noise)")
        
        print("\n   --- Filtered Out Candidates (Why they were rejected) ---")
        for i, dropped_doc in enumerate(rerank["dropped_documents"][:3]):
            print(f"   [x] Score: {dropped_doc['score']:6.2f} | Page {dropped_doc.get('page', '?')}: {dropped_doc['snippet']}")

        # 3. Context Window Injection Order
        print(f"\n[3. Lost-in-the-Middle Context Window Layout]")
        print("   (Position 1 = Top of Prompt | Last Position = Bottom near LLM Query)")
        for item in t["context_window_layout"]:
            print(f"   Pos #{item['position']} [Page {item['page']}]: {item['snippet']}")

        print("=" * 70 + "\n")