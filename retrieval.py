import numpy as np
import faiss
from typing import List, Tuple
from indexing import HybridIndexer


class HybridRetriever:
    def __init__(self, indexer: HybridIndexer):
        self.indexer = indexer
        self.dense_encoder = indexer.dense_encoder

    def retrieve(self,query:str,top_k:int=10,rrf_k:int=60):

        # SPARSE
        tokenized_query = query.lower().split()
        bm25_scores = self.indexer.bm25_index.get_scores(tokenized_query)
        sparse_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Dense
        query_vector = self.dense_encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        print(self.indexer)
        distances, dense_top_indices_2d = self.indexer.faiss_index.search(query_vector, top_k)
        dense_top_indices = dense_top_indices_2d[0]


        #Rrf score
        rrf_scores = {}

        for rank, doc_idx in enumerate(sparse_top_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc_idx in enumerate(dense_top_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank + 1))


        sorted_indices = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        fused_documents = []
        for doc_idx, score in sorted_indices[:top_k]:
            fused_documents.append(self.indexer.documents[doc_idx])

        return fused_documents
