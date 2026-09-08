from sentence_transformers import CrossEncoder


class DocumentReranker:
    def __init__(self,model_name:str="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.reranker=CrossEncoder(model_name)

    def rerank(self,query:str,documents,top_k:int=3,score_dropoff:float=3.0):
        if not documents:
            return []
        sentence_pairs=[[query,doc.page_content] for doc in documents]
        scores=self.reranker.predict(sentence_pairs)
        scored_docs=list(zip(scores,documents))
        scored_docs.sort(key=lambda x:x[0],reverse=True)
        top_score = scored_docs[0][0]


        filtered_doc=[]  
        for score,doc in scored_docs:
            if score>=(top_score - score_dropoff):
                filtered_doc.append(doc)
            else:
                break

        best_documents=filtered_doc[:top_k]
        print(f" -> Dynamic Threshold kept {len(best_documents)} highly relevant chunks.")
        recorded_docs=[]
        for i,doc in enumerate(best_documents):
            if i%2==1:
                recorded_docs.insert(0,doc)
            else:
                recorded_docs.append(doc)
        
        print(f" -> Re-ranking complete. Top score: {top_score:.4f}")
        
        return recorded_docs

