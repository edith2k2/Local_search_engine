from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer
import asyncio
from anthropic import AsyncAnthropic
import json
from nltk.tokenize import word_tokenize
import logging
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a single search result with its relevance information"""
    chunk_id: int
    text: str
    context: Optional[str]
    score: float
    source: str
    reasoning: Optional[str] = None

@dataclass
class RetrievalStep:
    """Represents a single step in the chain-of-thought retrieval process"""
    query: str
    reasoning: str
    results: List[SearchResult]
    next_query: Optional[str] = None

class ChainOfThoughtRetriever:
    """
    Implements chain-of-thought guided retrieval combining sparse and dense methods
    with LLM-based reasoning for iterative search refinement.
    """
    
    def __init__(
        self,
        documents: Dict[str, Dict],
        embedding_model: SentenceTransformer,
        anthropic_client: AsyncAnthropic,
        max_steps: int = 3,
        results_per_step: int = 5,
        device: str = None
    ):
        self.documents = documents
        self.embedding_model = embedding_model
        self.client = anthropic_client
        self.max_steps = max_steps
        self.results_per_step = results_per_step
        
        # Set device for computations
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model.to(self.device)
        
        # Initialize document indices
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize by keeping track of individual document indices and their BM25 indices"""
        # Initialize our tracking structures
        self.all_chunks = []          # Stores all chunk information
        self.doc_indices = {}         # Maps document paths to their FAISS indices
        self.bm25_indices = {}        # Maps document paths to their BM25 indices
        self.chunk_to_doc = {}        # Maps chunk IDs to their source documents
        
        current_chunk_id = 0
        
        # Process each document one at a time
        for doc_path, doc_data in self.documents.items():
            chunks = doc_data['chunks']
            
            # Store both FAISS and BM25 indices for this document
            self.doc_indices[doc_path] = doc_data['faiss_index']
            self.bm25_indices[doc_path] = doc_data['bm25_index']
            
            # Process each chunk in the document
            for chunk in chunks:
                chunk_id = current_chunk_id
                
                self.all_chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk['text'],
                    'context': chunk.get('context'),
                    'source': doc_path,
                    'original_chunk_id': chunk['chunk_id']
                })
                
                self.chunk_to_doc[chunk_id] = doc_path
                current_chunk_id += 1
    
    async def _get_dense_results(self, query: str, k: int) -> List[SearchResult]:
        """Get results by searching each document's index separately"""
        # First, we create the query embedding once - we'll use this for all searches
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            device=self.device
        ).reshape(1, -1).astype('float32')
        
        # We'll collect all results here
        all_results = []
        
        # Search through each document's index
        for doc_path, index in self.doc_indices.items():
            # Search this document's FAISS index
            scores, indices = index.search(query_embedding, k)
            
            # Process the results from this document
            for score, idx in zip(scores[0], indices[0]):
                # Get the chunk information
                chunk = self.all_chunks[idx]
                
                # Verify we're mapping to the correct document
                # This is a safety check to ensure our indexing is correct
                if chunk['source'] == doc_path:
                    all_results.append(SearchResult(
                        chunk_id=chunk['chunk_id'],
                        text=chunk['text'],
                        context=chunk['context'],
                        score=-score,  # Convert distance to similarity score
                        source=doc_path
                    ))
        
        # Sort all results by score and return the top k
        return sorted(all_results, key=lambda x: x.score, reverse=True)[:k]
    
    def _get_sparse_results(self, query: str, k: int) -> List[SearchResult]:
        """Get results using sparse retrieval with pre-computed BM25 indices"""
        # Tokenize the query just like we did during preprocessing
        query_tokens = word_tokenize(query.lower())
        
        # We'll collect all results here
        all_results = []
        
        # Search through each document's BM25 index
        for doc_path, bm25_index in self.bm25_indices.items():
            # Get BM25 scores for this document
            scores = bm25_index.get_scores(query_tokens)
            
            # Get top k indices for this document
            top_k_indices = np.argsort(scores)[-k:][::-1]
            
            # Convert document-specific indices to results
            for idx in top_k_indices:
                # Find the corresponding chunk in our global collection
                chunk = next(
                    chunk for chunk in self.all_chunks 
                    if chunk['source'] == doc_path and chunk['original_chunk_id'] == idx
                )
                
                all_results.append(SearchResult(
                    chunk_id=chunk['chunk_id'],
                    text=chunk['text'],
                    context=chunk['context'],
                    score=scores[idx],
                    source=doc_path
                ))
        
        # Sort by score and return top k overall
        return sorted(all_results, key=lambda x: x.score, reverse=True)[:k]
    
    def _merge_results(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int
    ) -> List[SearchResult]:
        """Merge results from dense and sparse retrieval using rank fusion"""
        # Create score dictionaries
        dense_scores = {r.chunk_id: (i + 1, r) for i, r in enumerate(dense_results)}
        sparse_scores = {r.chunk_id: (i + 1, r) for i, r in enumerate(sparse_results)}
        
        # Compute reciprocal rank fusion scores
        fusion_scores = {}
        for chunk_id in set(dense_scores.keys()) | set(sparse_scores.keys()):
            dense_rank = dense_scores.get(chunk_id, (len(dense_results) + 1, None))[0]
            sparse_rank = sparse_scores.get(chunk_id, (len(sparse_results) + 1, None))[0]
            
            # RRF formula with k=60 (default constant)
            fusion_scores[chunk_id] = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)
        
        # Sort by fusion score and get top k
        top_chunks = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Create merged results list
        merged_results = []
        for chunk_id, fusion_score in top_chunks:
            # Get the result object from either dense or sparse results
            result = (dense_scores.get(chunk_id, (None, None))[1] or 
                     sparse_scores.get(chunk_id, (None, None))[1])
            result.score = fusion_score  # Update score to fusion score
            merged_results.append(result)
        
        return merged_results
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_llm_reasoning(
        self,
        query: str,
        current_results: List[SearchResult],
        previous_steps: List[RetrievalStep]
    ) -> Tuple[str, Optional[str]]:
        """Get LLM reasoning about search results and next query refinement"""
        try:
            # Construct context from previous steps if they exist
            previous_context = ""
            if previous_steps:
                step_summaries = []
                for i, step in enumerate(previous_steps, 1):
                    first_line_summary = step.reasoning.split('\n')[0]
                    summary = f"""Step {i}:
    - Query: "{step.query}"
    - Key Findings: {first_line_summary}  
    """
                    step_summaries.append(summary)
                previous_context = "Previous Search History:\n" + "\n".join(step_summaries)

            # Format current results with clear structure
            results_context = "Current Search Results:\n"
            for i, result in enumerate(current_results, 1):
                # Include source and relevance score for better context
                results_context += f"""Result {i} (Source: {Path(result.source).name}, Score: {result.score:.2f}):
    - Content: {result.text[:300]}...
    """

            # Construct the structured analysis prompt
            prompt = f"""Analyze these search results for the query: "{query}"

    {previous_context}

    {results_context}

    Step-by-step Analysis:

    1. Core Information Found:
    - What specific information in these results directly answers the query?
    - What are the key concepts or topics covered?
    - How relevant are the top results to the query?

    2. Missing Elements:
    - What specific aspects of the query are not addressed in these results?
    - What important related information would provide more complete understanding?
    - Are there any gaps in the current coverage?

    3. Query Refinement Decision:
    Based on this analysis:
    a) If the results provide comprehensive coverage:
        Explain why no refinement is needed.
    
    b) If important information is missing:
        - Identify the specific gap to target
        - Construct a precise query focused on that gap
        - Use relevant terms found in current results
        - Target unexplored aspects of the topic

    Provide your complete analysis and then clearly indicate either:
    SUFFICIENT: [Explanation of why results are complete]
    or
    REFINED QUERY: [New specific query targeting identified gaps]
    """

            # Get LLM response with error handling
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=750,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            
            llm_response = response.content[0].text.strip()
            
            # Parse the LLM response to extract reasoning and potential refined query
            reasoning = llm_response
            refined_query = None
            
            # Check if the response contains a refined query
            if "REFINED QUERY:" in llm_response:
                parts = llm_response.split("REFINED QUERY:", 1)
                reasoning = parts[0].strip()
                refined_query = parts[1].strip()
            
            # Log the analysis for debugging and improvement
            logger.debug(f"Query Analysis:\nOriginal Query: {query}\nReasoning: {reasoning}\n"
                        f"Refined Query: {refined_query}")
            
            return reasoning, refined_query
            
        except Exception as e:
            logger.error(f"Error in LLM reasoning: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Results count: {len(current_results)}")
            raise
    
    async def search(
        self,
        query: str,
        return_reasoning: bool = True
    ) -> Tuple[List[SearchResult], List[RetrievalStep]]:
        """
        Perform chain-of-thought guided retrieval for the given query.
        Returns final results and reasoning steps if requested.
        """
        current_query = query
        retrieval_steps = []
        
        for step in range(self.max_steps):
            # Get results from both retrieval methods
            dense_results = await self._get_dense_results(
                current_query,
                self.results_per_step
            )

            sparse_results = self._get_sparse_results(
                current_query,
                self.results_per_step
            )
            
            # Merge results
            merged_results = self._merge_results(
                dense_results,
                sparse_results,
                self.results_per_step
            )
            
            # Get LLM reasoning
            # reasoning, next_query = await self._get_llm_reasoning(
            #     current_query,
            #     merged_results,
            #     retrieval_steps
            # )
            reasoning, next_query = "testing mode", None
            
            # Record this step
            current_step = RetrievalStep(
                query=current_query,
                reasoning=reasoning,
                results=merged_results,
                next_query=next_query
            )
            retrieval_steps.append(current_step)
            
            # Stop if no next query is suggested
            if not next_query:
                break
                
            current_query = next_query
        
        # Combine and re-rank all results from all steps
        all_results = []
        seen_chunks = set()
        for step in retrieval_steps:
            for result in step.results:
                if result.chunk_id not in seen_chunks:
                    result.reasoning = step.reasoning
                    all_results.append(result)
                    seen_chunks.add(result.chunk_id)
        
        # Sort by score (could implement more sophisticated final ranking)
        final_results = sorted(all_results, key=lambda x: x.score, reverse=True)
        
        if return_reasoning:
            return final_results, retrieval_steps
        return final_results, []

    async def search_with_feedback(
        self,
        query: str,
        relevance_feedback: Optional[Dict[int, float]] = None
    ) -> Tuple[List[SearchResult], List[RetrievalStep]]:
        """
        Perform search with optional relevance feedback from previous results.
        relevance_feedback should be a dict mapping chunk_ids to relevance scores.
        """
        # TODO: Implement relevance feedback mechanism
        pass