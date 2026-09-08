import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

class HybridIndexer:
    def __init__(self, dense_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.dense_encoder = SentenceTransformer(dense_model_name)
        self.faiss_index = None
        self.bm25_index = None
        self.documents = [] 

    def build_indexes(self, documents):
        self.documents = documents
        
        texts = [doc.page_content for doc in documents]
        
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_texts)

        dense_embedding = self.dense_encoder.encode(texts, convert_to_numpy=True)
        dimension = dense_embedding.shape[1]
        
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(dense_embedding)
        
        self.faiss_index.add(dense_embedding) 

    def save_indexes(self, save_dir: str = "db/"):
        os.makedirs(save_dir, exist_ok=True)
        
        faiss.write_index(self.faiss_index, os.path.join(save_dir, "dense.faiss"))
        
        with open(os.path.join(save_dir, "sparse_and_docs.pkl"), "wb") as f:
            pickle.dump({"bm25": self.bm25_index, "docs": self.documents}, f)

    def load_indexes(self, save_dir: str = "db/"):
    
        self.faiss_index = faiss.read_index(os.path.join(save_dir, "dense.faiss"))

        
        with open(os.path.join(save_dir, "sparse_and_docs.pkl"), "rb") as f: 
            data = pickle.load(f)
            self.bm25_index = data["bm25"]
            self.documents = data['docs']
            
            
        print("Indexes loaded successfully.")