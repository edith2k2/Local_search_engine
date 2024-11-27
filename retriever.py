from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import util
import torch
from nltk.tokenize import word_tokenize
from dataclasses import dataclass
from anthropic import AsyncAnthropic
import asyncio
from pydantic import BaseModel

@dataclass
class SearchResult:
    chunk_id: int
    file_path: str
    text: str
    context: str
    score: float
    retrieval_method: str

class SearchStep(BaseModel):
    reasoning: str
    sub_query: str
    results: List[SearchResult]

class DocumentRetriever:
    def __init__(
        self,
        documents: Dict[str, Dict],
        anthropic_client: AsyncAnthropic,
        bm25_weight: float = 0.3,
        embedding_weight: float = 0.7,
        top_k: int = 5,
        max_search_steps: int = 3  # Limit the number of search iterations
    ):
        self.documents = documents
        self.client = anthropic_client
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.top_k = top_k
        self.max_search_steps = max_search_steps

    async def iterative_search(self, query: str) -> Tuple[List[SearchResult], List[SearchStep]]:
        """
        Perform an iterative search guided by LLM reasoning.
        Returns both final results and the search steps for transparency.
        """
        search_history = []
        all_results = []
        
        # Initial search planning
        planning_prompt = f"""You are helping with a document search strategy. 
        Query: "{query}"
        
        Think step by step about how to break this search into sub-queries. Consider:
        1. What's the core information need?
        2. What related aspects might need exploration?
        3. How should we start the search?
        
        Provide the first focused sub-query we should try. Format:
        Reasoning: <your step-by-step thought process>
        Sub-query: <specific search query to try>
        
        Be direct and clear."""

        response = await self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            temperature=0.3,
            messages=[{"role": "user", "content": planning_prompt}]
        )
        
        # Extract initial reasoning and sub-query
        llm_output = response.content[0].text.strip()
        reasoning = llm_output.split("Sub-query:")[0].replace("Reasoning:", "").strip()
        sub_query = llm_output.split("Sub-query:")[1].strip()
        
        # Iterative search loop
        for step in range(self.max_search_steps):
            # Perform search with current sub-query
            current_results = await self.hybrid_search(sub_query)
            
            # Record this search step
            search_step = SearchStep(
                reasoning=reasoning,
                sub_query=sub_query,
                results=current_results
            )
            search_history.append(search_step)
            
            # Update all_results
            all_results.extend(current_results)
            
            # Get context from top results
            context_text = "\n".join([
                f"Result {i+1}:\n{result.text}\n{result.context}"
                for i, result in enumerate(current_results[:3])
            ])
            
            # Ask LLM to evaluate and plan next step
            evaluation_prompt = f"""Original query: "{query}"
            
            Search history:
            {'\n'.join(f'Step {i+1}: {step.sub_query} - {step.reasoning}' 
                      for i, step in enumerate(search_history))}
            
            Recent results:
            {context_text}
            
            Evaluate the search progress and decide next step:
            1. Have we found enough information to answer the original query?
            2. If not, what aspect still needs exploration?
            3. How should we adjust our search?

            Format response as:
            Status: [COMPLETE/CONTINUE]
            Reasoning: <your analysis>
            Next-query: <next search query if status is CONTINUE>
            
            Be precise and direct."""

            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            
            eval_output = response.content[0].text.strip()
            status = eval_output.split("Status:")[1].split("\n")[0].strip()
            
            if "COMPLETE" in status or step == self.max_search_steps - 1:
                break
                
            # Extract reasoning and next query
            reasoning = eval_output.split("Reasoning:")[1].split("Next-query:")[0].strip()
            sub_query = eval_output.split("Next-query:")[1].strip()
        
        # Final ranking of all results
        final_results = self.rank_fusion(all_results)
        
        return final_results[:self.top_k], search_history

    async def hybrid_search(self, query: str) -> List[SearchResult]:
        """Combine BM25 and embedding search results."""
        bm25_results = self.bm25_search(query)
        embedding_results = await self.embedding_search(query)
        
        # Combine results using rank fusion
        combined_results = self.rank_fusion(bm25_results + embedding_results)
        return combined_results

    def bm25_search(self, query: str) -> List[SearchResult]:
        """Perform BM25 search across all documents."""
        query_tokens = word_tokenize(query.lower())
        all_results = []
        
        for file_path, doc_data in self.documents.items():
            bm25 = doc_data['bm25_index']
            scores = bm25.get_scores(query_tokens)
            
            for chunk_id, score in enumerate(scores):
                chunk = doc_data['chunks'][chunk_id]
                if score > 0:
                    all_results.append(SearchResult(
                        chunk_id=chunk_id,
                        file_path=file_path,
                        text=chunk['text'],
                        context=chunk.get('context', ''),
                        score=score,
                        retrieval_method='bm25'
                    ))
        
        return sorted(all_results, key=lambda x: x.score, reverse=True)[:self.top_k]

    async def embedding_search(self, query: str) -> List[SearchResult]:
        """Perform embedding-based search across all documents."""
        embeddings = await self.process_embeddings([query])
        query_embedding = embeddings[0]
        
        all_results = []
        for file_path, doc_data in self.documents.items():
            chunk_embeddings = np.array([chunk['embedding'] for chunk in doc_data['chunks']])
            
            similarities = util.cos_sim(
                torch.tensor([query_embedding]),
                torch.tensor(chunk_embeddings)
            )[0].numpy()
            
            for chunk_id, score in enumerate(similarities):
                chunk = doc_data['chunks'][chunk_id]
                if score > 0:
                    all_results.append(SearchResult(
                        chunk_id=chunk_id,
                        file_path=file_path,
                        text=chunk['text'],
                        context=chunk.get('context', ''),
                        score=float(score),
                        retrieval_method='embedding'
                    ))
        
        return sorted(all_results, key=lambda x: x.score, reverse=True)[:self.top_k]

    def rank_fusion(self, results: List[SearchResult]) -> List[SearchResult]:
        """Combine and re-rank results using weighted reciprocal rank fusion."""
        chunk_scores = {}
        
        for rank, result in enumerate(results):
            chunk_key = (result.file_path, result.chunk_id)
            rr = 1 / (rank + 1)
            weight = self.bm25_weight if result.retrieval_method == 'bm25' else self.embedding_weight
            weighted_score = rr * weight
            
            if chunk_key not in chunk_scores:
                chunk_scores[chunk_key] = {
                    'result': result,
                    'score': weighted_score
                }
            else:
                chunk_scores[chunk_key]['score'] += weighted_score
        
        final_results = []
        for chunk_key, data in chunk_scores.items():
            result = data['result']
            result.score = data['score']
            final_results.append(result)
        
        return sorted(final_results, key=lambda x: x.score, reverse=True)

    async def process_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Compute embeddings for search queries."""
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        try:
            with torch.no_grad():
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    device=device,
                    batch_size=32
                )
            return embeddings
        except Exception as e:
            print(f"Error computing embeddings: {str(e)}")
            return [np.zeros(model.get_sentence_embedding_dimension()) for _ in texts]
        finally:
            if device == 'cuda':
                torch.cuda.empty_cache()