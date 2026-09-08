import os
import time

# 1. Core Pipeline Imports
from ingestion import DocumentLoader
from indexing import HybridIndexer
from retrieval import HybridRetriever
from reranker import DocumentReranker
from generator import LocalLLMGenerator

# Handle typo safety if your file/class is named RedisCache or RedisCahce
try:
    from redis_cache import RedisCache
except ImportError:
    from redis_cache import RedisCahce as RedisCache

# 2. Advanced Feature Imports
from query_transformation import QueryTransformer
from eval import run_evaluation
from debugger import RAGDebugger


def run_chat_mode(transformer, retriever, reranker, generator):
    """Launches the interactive terminal loop with on-demand Diagnostic Inspection."""
    print("\n==================================================")
    print("      ESP32 HARDWARE RAG INTELLIGENCE TERMINAL     ")
    print("==================================================")
    print("Type 'exit' or 'quit' to terminate the session.\n")

    query_cache = RedisCache()
    debugger = RAGDebugger()

    while True:
        user_query = input("[Engineer] Ask a hardware question: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ['exit', 'quit']:
            print("\nShutting down terminal. Goodbye.")
            break

        start_time = time.time()

        # ---------------------------------------------------------
        # Cache Check
        # ---------------------------------------------------------
        cached_answer = query_cache.check_cache(user_query)
        if cached_answer:
            print("\n[CACHE HIT] Answer retrieved instantly from Redis memory.")
            print("=== FINAL ANSWER ===")
            print(cached_answer)
            print("====================")
            print(f"Latency: {time.time() - start_time:.4f} seconds")
            print(f"Cache Size: {query_cache.get_len()} items stored.")
            continue

        print("\n[Cache Miss] Processing through RAG Pipeline...")
        print("--- Pipeline Execution Progress ---")

        # Initialize debugger trace for this question
        debugger.start_trace(user_query)
        
        # Step 1: Query Transformation
        print("1. Optimizing engineering input...")
        optimized_query = transformer.rewrite_query(user_query)
        print(f"   [Raw] -> {user_query}")
        print(f"   [Optimized] -> {optimized_query}")
        
        debugger.log_routing(transformed_query=optimized_query, was_routed=True)
        
        # Step 2: Wide Net Retrieval
        print("2. Querying FAISS and BM25 Indexes...")
        hybrid_candidates = retriever.retrieve(optimized_query, top_k=15)
        
        # Step 3: Precision Re-ranking & Score Collection
        print("3. Scoring candidates with Cross-Encoder...")
        
        # Compute raw scores for diagnostics before filtering
        sentence_pairs = [[optimized_query, doc.page_content] for doc in hybrid_candidates]
        raw_scores = reranker.reranker.predict(sentence_pairs)

        scored_items = [
            {
                "score": float(score),
                "page": doc.metadata.get("page", "N/A"),
                "snippet": doc.page_content[:90].replace("\n", " ").strip() + "..."
            }
            for score, doc in sorted(zip(raw_scores, hybrid_candidates), key=lambda x: x[0], reverse=True)
        ]

        best_docs = reranker.rerank(optimized_query, hybrid_candidates, top_k=3)

        # Log scoring, dynamic thresholding, and prompt injection layout
        debugger.log_reranking(
            scored_docs_with_metadata=scored_items,  
            threshold_cutoff=3.0,
            top_k_docs=best_docs,                    
            reordered_docs=best_docs
        )
        
        # Step 4: Grounded Response Generation
        print("4. Executing Guided Generation...")
        answer = generator.generate_answer(optimized_query, best_docs)

        # Step 5: Storing in Cache
        query_cache.save_to_cache(user_query, answer)

        total_latency = time.time() - start_time

        # Display Final Output
        print("\n=== FINAL ANSWER ===")
        print(answer)
        print("====================")
        print(f"Latency: {total_latency:.2f} seconds")
        print(f"Cache Size: {query_cache.get_len()} items stored.")

        # ---------------------------------------------------------
        # Interactive Diagnostic Report Prompt
        # ---------------------------------------------------------
        show_debug = input("\n[Diagnostic] View diagnostic trace report for this query? (y/n): ").strip().lower()
        if show_debug in ['y', 'yes']:
            debugger.print_diagnostic_report()


def main():
    PDF_PATH = "data/esp32_technical_refrence_manual_en.pdf"
    DB_PATH = "db/"
    
    print("=== Booting Advanced Hardware RAG System ===")

    # ==========================================
    # A. INITIALIZE & LOAD DATABASES
    # ==========================================
    indexer = HybridIndexer()
    if os.path.exists(os.path.join(DB_PATH, "dense.faiss")):
        print("Found existing database. Loading...")
        indexer.load_indexes(save_dir=DB_PATH)
    else:
        print("No database found. Processing PDF...")
        loader = DocumentLoader(PDF_PATH)
        documents = loader.loader()
        indexer.build_indexes(documents)
        indexer.save_indexes(save_dir=DB_PATH)

    # ==========================================
    # B. INSTANTIATE CORE MODULES
    # ==========================================
    retriever = HybridRetriever(indexer=indexer)
    reranker = DocumentReranker()
    generator = LocalLLMGenerator(model_name="openai/gpt-oss-120b")
    transformer = QueryTransformer(model_name="openai/gpt-oss-120b")

    # ==========================================
    # C. EXECUTION ROUTING
    # ==========================================
    print("\nSelect Application Execution Mode:")
    print("1: Interactive Hardware Assistant Terminal (Chat Mode)")
    print("2: Performance Benchmarking Suite (Evaluation Mode)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_chat_mode(transformer, retriever, reranker, generator)
    elif choice == "2":
        run_evaluation(retriever, reranker, generator)
    else:
        print("[Invalid Selection] Terminating execution pipeline workflow.")


if __name__ == "__main__":
    main()