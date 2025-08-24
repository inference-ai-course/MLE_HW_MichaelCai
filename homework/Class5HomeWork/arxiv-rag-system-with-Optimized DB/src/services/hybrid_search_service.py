from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from .faiss_service import FaissService
from .database_service import DatabaseService
from .openai_embedding_service import OpenAIEmbeddingService


class HybridSearchService:
    def __init__(self, faiss_service: FaissService, database_service: DatabaseService, 
                 embedding_service: OpenAIEmbeddingService):
        self.faiss_service = faiss_service
        self.database_service = database_service
        self.embedding_service = embedding_service
        self.bm25 = None
        self.documents = []
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 with all documents from FAISS."""
        if hasattr(self.faiss_service, 'document_store') and self.faiss_service.document_store:
            self.documents = [doc['text'] for doc in self.faiss_service.document_store]
            tokenized_docs = [doc.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _get_bm25_scores(self, query: str) -> Dict[str, float]:
        """Get BM25 scores for documents."""
        if not self.bm25 or not self.documents:
            return {}
        
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        normalized_scores = self._normalize_scores(scores.tolist())
        
        # Create mapping from document text to score
        doc_scores = {}
        for doc, score in zip(self.documents, normalized_scores):
            doc_scores[doc] = score
        
        return doc_scores
    
    def _get_faiss_results(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Get FAISS vector search results with normalized scores."""
        query_embedding = self.embedding_service.embed_query(query)
        results = self.faiss_service.search(query_embedding, k=limit)
        
        faiss_results = []
        scores = []
        
        for doc in results:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            # FAISS returns similarity scores, we need to extract them
            score = doc.metadata.get('score', 0.0) if hasattr(doc, 'metadata') else 0.0
            faiss_results.append((content, score))
            scores.append(score)
        
        # Normalize FAISS scores
        if scores:
            normalized_scores = self._normalize_scores(scores)
            faiss_results = [(content, norm_score) for (content, _), norm_score 
                           in zip(faiss_results, normalized_scores)]
        
        return faiss_results
    
    def _get_fts5_results(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Get FTS5 search results with normalized scores."""
        results = self.database_service.search_documents(query, limit)
        
        fts5_results = []
        scores = []
        
        for result in results:
            content = result.get('content', '')
            # FTS5 rank is negative (lower is better), convert to positive score
            rank = result.get('rank', 0)
            score = abs(rank) if rank != 0 else 0.0
            fts5_results.append((content, score))
            scores.append(score)
        
        # Normalize FTS5 scores (invert since lower rank is better)
        if scores:
            # Invert scores so higher is better
            max_score = max(scores) if scores else 1.0
            inverted_scores = [max_score - score + 1 for score in scores]
            normalized_scores = self._normalize_scores(inverted_scores)
            fts5_results = [(content, norm_score) for (content, _), norm_score 
                          in zip(fts5_results, normalized_scores)]
        
        return fts5_results
    
    def hybrid_score(self, vec_score: float, keyword_score: float, alpha: float = 0.6) -> float:
        """Combine vector and keyword scores using weighted sum."""
        return alpha * vec_score + (1 - alpha) * keyword_score
    
    def search(self, query: str, limit: int = 10, alpha: float = 0.6) -> List[Dict]:
        """
        Perform hybrid search combining FAISS vector search and keyword search.
        
        Args:
            query: Search query
            limit: Number of results to return
            alpha: Weight for vector score (0-1). Higher = more weight to vector search
        
        Returns:
            List of documents with combined scores
        """
        # Get results from different search methods
        faiss_results = self._get_faiss_results(query, limit * 2)  # Get more to have overlap
        bm25_scores = self._get_bm25_scores(query)
        fts5_results = self._get_fts5_results(query, limit * 2)
        
        # Combine all unique documents
        all_docs = {}
        
        # Add FAISS results
        for content, vec_score in faiss_results:
            all_docs[content] = {
                'content': content,
                'vector_score': vec_score,
                'bm25_score': bm25_scores.get(content, 0.0),
                'fts5_score': 0.0
            }
        
        # Add FTS5 results
        for content, fts5_score in fts5_results:
            if content in all_docs:
                all_docs[content]['fts5_score'] = fts5_score
            else:
                all_docs[content] = {
                    'content': content,
                    'vector_score': 0.0,
                    'bm25_score': bm25_scores.get(content, 0.0),
                    'fts5_score': fts5_score
                }
        
        # Calculate hybrid scores
        combined_results = []
        for doc_data in all_docs.values():
            # Combine BM25 and FTS5 as keyword score
            keyword_score = max(doc_data['bm25_score'], doc_data['fts5_score'])
            
            combined_score = self.hybrid_score(
                doc_data['vector_score'], 
                keyword_score, 
                alpha
            )
            
            combined_results.append({
                'content': doc_data['content'],
                'combined_score': combined_score,
                'vector_score': doc_data['vector_score'],
                'keyword_score': keyword_score,
                'bm25_score': doc_data['bm25_score'],
                'fts5_score': doc_data['fts5_score']
            })
        
        # Sort by combined score and return top results
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return combined_results[:limit]