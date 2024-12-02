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
import time
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from query_classifier import QueryClassifier, QueryType
from query_parser import TemporalQueryParser, TemporalConstraints, TimeFrame, SearchParameters
from dataclasses import field

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Provides a clean, formatted string representation"""
        source_name = Path(self.source).name
        text_preview = f"{self.text[:50]}..." if len(self.text) > 50 else self.text
        
        output = [
            f"\nSearch Result #{self.chunk_id}",
            f"Score: {self.score:.3f}",
            f"Source: {source_name}",
            f"Text: {text_preview}"
        ]
        
        if self.context:
            context_preview = self.context
            output.append(f"Context: {context_preview}")
            
        if self.reasoning:
            reasoning_preview = self.reasoning
            output.append(f"Reasoning: {reasoning_preview}")
            
        return "\n".join(output)

    __repr__ = __str__

@dataclass
class ReasoningStep:
    """Represents a single step in the chain-of-thought reasoning process"""
    relevance_findings: Dict[str, float]
    gaps_identified: List[str]
    redundant_content: List[Tuple[str, str]]
    suggested_refinement: Optional[str]
    reasoning_explanation: str
    confidence_score: float

    def __str__(self) -> str:
        """Provides a clean, formatted string representation"""
        output = ["\nReasoning Analysis"]
        output.append("=" * 20)
        
        output.append("\nConfidence Score: {:.2f}".format(self.confidence_score))
        
        output.append("\nRelevance Findings:")
        for result_id, score in self.relevance_findings.items():
            output.append(f"  • {result_id}: {score:.2f}")
        
        output.append("\nIdentified Gaps:")
        for gap in self.gaps_identified:
            output.append(f"  • {gap}")
        
        output.append("\nRedundant Content:")
        for pair in self.redundant_content:
            output.append(f"  • Results {pair[0]} and {pair[1]} overlap")
        
        if self.suggested_refinement:
            output.append(f"\nSuggested Refinement: {self.suggested_refinement}")
        
        output.append("\nReasoning:")
        reasoning_preview = (f"{self.reasoning_explanation[:100]}..." 
                           if len(self.reasoning_explanation) > 100 
                           else self.reasoning_explanation)
        output.append(f"  {reasoning_preview}")
        
        return "\n".join(output)

    __repr__ = __str__

@dataclass
class SearchIteration:
    """Tracks a single iteration of the search process"""
    query: str
    results: List[SearchResult]
    reasoning: ReasoningStep
    combined_scores: Dict[str, float]
    timestamp: float

    def __str__(self) -> str:
        """Provides a clean, formatted string representation"""
        time_str = datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        output = [
            "\nSearch Iteration",
            "=" * 50,
            f"\nTimestamp: {time_str}",
            f"Query: {self.query}"
        ]
        
        output.append("\nResults Summary:")
        output.append(f"Total Results: {len(self.results)}")
        if self.results:
            output.append("Top Results:")
            for result in sorted(self.results, key=lambda x: x.score, reverse=True)[:3]:
                output.append(str(result))
        
        output.append("\nCombined Scores:")
        for chunk_id, score in self.combined_scores.items():
            output.append(f"  • Chunk {chunk_id}: {score:.2f}")
        
        output.append("\nReasoning Analysis:")
        output.append(str(self.reasoning))
        
        return "\n".join(output)

    __repr__ = __str__

class ChainOfThoughtRetriever:
    """
    Implements an enhanced retrieval system combining dense and sparse search with 
    chain-of-thought reasoning for iterative refinement.
    """
    
    def __init__(
        self,
        documents: Dict[str, Dict],
        embedding_model: SentenceTransformer,
        anthropic_client: AsyncAnthropic,
        device: str = None,
        max_iterations: int = 3,
        min_confidence_threshold: float = 0.7,
        results_per_step: int = 5
    ):
        """
        Initialize the retriever with necessary components and configuration.
        
        Args:
            documents: Dictionary mapping document paths to their processed content
            embedding_model: SentenceTransformer model for dense retrieval
            anthropic_client: Claude API client for reasoning
            device: Computing device for embeddings
            max_iterations: Maximum number of refinement iterations
            min_confidence_threshold: Minimum confidence to accept results
            results_per_step: Number of results to return per iteration
        """
        self.documents = documents
        self.embedding_model = embedding_model
        self.client = anthropic_client
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_iterations = max_iterations
        self.min_confidence_threshold = min_confidence_threshold
        self.results_per_step = results_per_step
        self.query_classifier = QueryClassifier()
        self.temporal_parser = TemporalQueryParser()
        
        # Initialize document indices
        self._initialize_indices()
        
        # Initialize prompts
        self._init_prompts()
    
    def _initialize_indices(self):
        """Initialize by organizing document indices and content for efficient retrieval"""
        # Initialize our tracking structures
        self.all_chunks = []  # Stores all chunk information
        self.doc_indices = {}  # Maps document paths to their FAISS indices
        self.bm25_indices = {}  # Maps document paths to their BM25 indices
        self.chunk_to_doc = {}  # Maps chunk IDs to their source documents
        
        current_chunk_id = 0
        
        # Process each document one at a time
        for doc_path, doc_data in self.documents.items():
            chunks = doc_data['chunks']
            metadata = doc_data['metadata']

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
                    'original_chunk_id': chunk['chunk_id'],
                    'metadata': metadata
                })
                
                self.chunk_to_doc[chunk_id] = doc_path
                current_chunk_id += 1
    
    def _init_prompts(self):
        """Initialize structured prompts for different reasoning stages"""
        self.analysis_prompt_template = """Analyze these search results for the query: "{query}"

1. **Relevance Analysis**:
    - Which results directly address the query? List result IDs and explain their relevance.
    - Are the top results relevant? If not, explain why.
    Score each result's relevance from 0-1 and justify your scoring.

2. **Coverage Analysis**:
    - What specific information is missing from these results?
    - What additional topics or concepts would enhance understanding?
    - Identify any gaps between the query's requirements and the available information.

3. **Redundancy Check**:
    - Identify any results that cover the same information.
    - Suggest which redundant results to prioritize and why.

4. **Query Refinement Decision**:
    Consider:
    - The gaps identified above
    - The relevance of current results
    - The redundancy in the information
    
    Then either:
    a) Suggest a specific refined query to address gaps and reduce redundancy
    OR
    b) Explain why the current results are sufficient

Format your response exactly as follows:
RELEVANCE_FINDINGS:
[List each result with its relevance score and justification]

GAPS_IDENTIFIED:
[List specific missing information or concepts]

REDUNDANT_CONTENT:
[List pairs of redundant results with preference ordering]

REFINEMENT_DECISION:
[Either "SUFFICIENT: [explanation]" or "REFINED_QUERY: [new query]"]

CONFIDENCE:
[Score 0-1 indicating confidence in this analysis]

REASONING:
[Detailed explanation of your analysis and decision]"""
    
    def update_documents(self, documents: Dict[str, Dict]):
        """
        Update the retriever by adding newly processed documents to existing indices.
        
        Args:
            documents: Dictionary mapping file paths to document data including
                    metadata, chunks, faiss_index, and bm25_index
        """
        try:
            # Get our current chunk_id to continue numbering
            current_chunk_id = len(self.all_chunks)
            
            # Process each new or updated document
            for doc_path, doc_data in documents.items():
                # Store indices for this document
                self.doc_indices[doc_path] = doc_data['faiss_index']
                self.bm25_indices[doc_path] = doc_data['bm25_index']
                
                # Process each chunk in the document
                for chunk in doc_data['chunks']:
                    chunk_id = current_chunk_id
                    
                    # Add to our chunk tracking structures
                    self.all_chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk['text'],
                        'context': chunk.get('context'),
                        'source': doc_path,
                        'original_chunk_id': chunk['chunk_id'],
                        'metadata': doc_data['metadata']
                    })
                    
                    self.chunk_to_doc[chunk_id] = doc_path
                    current_chunk_id += 1
            
            logger.info(
                f"Added {len(documents)} documents to retriever. "
                f"Total chunks now indexed: {len(self.all_chunks)}"
            )
            
        except Exception as e:
            logger.error(f"Error adding documents to retriever: {str(e)}")
            raise

    async def _get_dense_results(
        self,
        query: str,
        k: int
    ) -> List[SearchResult]:
        """
        Get results using dense retrieval with FAISS indices.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of search results with scores
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                device=self.device
            ).reshape(1, -1).astype('float32')
            
            # Collect results from all documents
            all_results = []
            
            # Search each document's index
            for doc_path, index in self.doc_indices.items():
                scores, indices = index.search(query_embedding, k)
                
                # Process results from this document
                for score, idx in zip(scores[0], indices[0]):
                    chunk = self.all_chunks[idx]
                    
                    if chunk['source'] == doc_path:
                        all_results.append(SearchResult(
                            chunk_id=chunk['chunk_id'],
                            text=chunk['text'],
                            context=chunk['context'],
                            score=-score,  # Convert distance to similarity
                            source=doc_path,
                            metadata=chunk['metadata']
                        ))
            
            # Sort by score and return top k
            return sorted(all_results, key=lambda x: x.score, reverse=True)[:k]
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {str(e)}")
            return []
    
    def _get_sparse_results(
        self,
        query: str,
        k: int
    ) -> List[SearchResult]:
        """
        Get results using sparse retrieval with BM25.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of search results with scores
        """
        try:
            # Tokenize query
            query_tokens = word_tokenize(query.lower())
            
            # Collect results from all documents
            all_results = []
            
            # Search each document's BM25 index
            for doc_path, bm25_index in self.bm25_indices.items():
                # Get BM25 scores
                scores = bm25_index.get_scores(query_tokens)
                
                # Get top k indices
                top_k_indices = np.argsort(scores)[-k:][::-1]
                
                # Convert to results
                for idx in top_k_indices:
                    chunk = next(
                        chunk for chunk in self.all_chunks 
                        if chunk['source'] == doc_path and 
                        chunk['original_chunk_id'] == idx
                    )
                    
                    all_results.append(SearchResult(
                        chunk_id=chunk['chunk_id'],
                        text=chunk['text'],
                        context=chunk['context'],
                        score=scores[idx],
                        source=doc_path,
                        metadata=chunk['metadata']
                    ))
            
            # Sort by score and return top k
            return sorted(all_results, key=lambda x: x.score, reverse=True)[:k]
            
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {str(e)}")
            return []
        
    def _get_fusion_k(self, query_type: QueryType) -> int:
        """Return appropriate k value based on query type"""
        k_values = {
            QueryType.FACTUAL: 40,      # More aggressive for fact-finding
            QueryType.REASONING: 80,     # More conservative for complex queries
            QueryType.COMPARISON: 60,    # Balanced for comparisons
            QueryType.EXPLORATORY: 100,  # Very conservative for exploration
            QueryType.PROCEDURAL: 50     # Moderately aggressive for how-to queries
        }
        return k_values.get(query_type, 60)  # Default to 60
    
    def _merge_results(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        weights: Dict[str, float],
        query_type: QueryType,
        k: int
    ) -> List[SearchResult]:
        """
        Merge results from dense and sparse retrieval using rank fusion.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: Number of results to return
            
        Returns:
            Merged and re-ranked results
        """
        # Create score dictionaries
        dense_scores = {r.chunk_id: (i + 1, r) for i, r in enumerate(dense_results)}
        sparse_scores = {r.chunk_id: (i + 1, r) for i, r in enumerate(sparse_results)}
        
        # Get query-appropriate k value
        fusion_k = self._get_fusion_k(query_type)
        logger.info(f"Using fusion k value: {fusion_k}")

        # Compute reciprocal rank fusion scores
        fusion_scores = {}
        for chunk_id in set(dense_scores.keys()) | set(sparse_scores.keys()):
            dense_rank = dense_scores.get(chunk_id, (len(dense_results) + 1, None))[0]
            sparse_rank = sparse_scores.get(chunk_id, (len(sparse_results) + 1, None))[0]
            
            # RRF formula with k=60 (default constant)
            fusion_scores[chunk_id] = (
                weights['dense'] * (1 / (fusion_k + dense_rank)) +
                weights['sparse'] * (1 / (fusion_k + sparse_rank))
            )
        
        # Sort by fusion score
        top_chunks = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Create merged results
        merged_results = []
        for chunk_id, fusion_score in top_chunks:
            # Get result object from either set
            result = (dense_scores.get(chunk_id, (None, None))[1] or 
                     sparse_scores.get(chunk_id, (None, None))[1])
            result.score = fusion_score
            merged_results.append(result)
        
        return merged_results

    def _check_redundancy(self, results: List[SearchResult]) -> List[Tuple[str, str]]:
        """
        Check for redundant content in results using semantic similarity.
        
        Args:
            results: List of search results to check
            
        Returns:
            List of redundant result pairs
        """
        redundant_pairs = []
        
        # Get embeddings for all results
        texts = [r.text for r in results]
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            device=self.device
        )
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find highly similar pairs
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if similarity_matrix[i][j] > 0.85:  # Similarity threshold
                    redundant_pairs.append((
                        str(results[i].chunk_id),
                        str(results[j].chunk_id)
                    ))
        
        return redundant_pairs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_reasoned_analysis(
        self,
        query: str,
        results: List[SearchResult],
        previous_steps: List[SearchIteration]
    ) -> ReasoningStep:
        """
        Get structured reasoning about search results from the LLM.
        
        Args:
            query: Current search query
            results: Current search results
            previous_steps: Previous search iterations
            
        Returns:
            Structured reasoning step
        """
        try:
            # Create previous steps context
            previous_context = ""
            if previous_steps:
                step_summaries = []
                for i, step in enumerate(previous_steps, 1):
                    summary = f"""Step {i}:
- Query: "{step.query}"
- Key Findings: {', '.join(step.reasoning.gaps_identified[:2])}
- Confidence: {step.reasoning.confidence_score:.2f}
"""
                    step_summaries.append(summary)
                previous_context = "Previous Steps:\n" + "\n".join(step_summaries)

            # Format current results
            results_context = "\n\n".join(
                f"""Result {i}:
Source: {Path(r.source).name}
Score: {r.score:.3f}

Content:
{r.text}

Context: {r.context or 'No additional context'}
---"""
                for i, r in enumerate(results, 1)
            )
            
            # Check for redundancy
            redundant_pairs = self._check_redundancy(results)
            
            # Construct the full prompt
            full_prompt = f"""{previous_context}

Current Query: "{query}"

Search Results:
{results_context}

Identified Redundancies:
{json.dumps(redundant_pairs, indent=2)}

{self.analysis_prompt_template.format(query=query)}"""

            # Get LLM response
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Parse structured response
            response_text = response.content[0].text.strip()
            
            # Extract sections
            sections = {}
            current_section = None
            current_content = []
            
            for line in response_text.split('\n'):
                if line.strip().endswith(':') and line.strip().upper() == line.strip():
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = line.strip()[:-1]
                    current_content = []
                else:
                    if current_section:
                        current_content.append(line)

            # Ensure we capture the last section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()

            # Parse relevance findings into a dictionary
            relevance_findings = {}
            for line in sections.get('RELEVANCE_FINDINGS', '').split('\n'):
                if 'Result' in line and ':' in line:
                    parts = line.split(':')
                    result_id = parts[0].strip()
                    score_text = parts[1].split()[0]
                    try:
                        score = float(score_text)
                        relevance_findings[result_id] = score
                    except ValueError:
                        continue

            # Parse redundant content into pairs
            redundant_pairs = []
            for line in sections.get('REDUNDANT_CONTENT', '').split('\n'):
                if '-' in line:
                    parts = line.strip().split('-')
                    if len(parts) == 2:
                        redundant_pairs.append(
                            (parts[0].strip(), parts[1].strip())
                        )

            # Extract refinement decision
            refinement_section = sections.get('REFINEMENT_DECISION', '')
            suggested_refinement = None
            if 'REFINED_QUERY:' in refinement_section:
                suggested_refinement = refinement_section.split('REFINED_QUERY:', 1)[1].strip()

            # Parse confidence score
            try:
                confidence = float(sections.get('CONFIDENCE', '0.5').strip())
            except ValueError:
                confidence = 0.5

            return ReasoningStep(
                relevance_findings=relevance_findings,
                gaps_identified=[gap.strip() for gap in sections.get('GAPS_IDENTIFIED', '').split('\n') if gap.strip()],
                redundant_content=redundant_pairs,
                suggested_refinement=suggested_refinement,
                reasoning_explanation=sections.get('REASONING', ''),
                confidence_score=confidence
            )

        except Exception as e:
            logger.error(f"Error in reasoned analysis: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Results count: {len(results)}")
            raise

    def _combine_scores(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Combine scores from dense and sparse retrieval methods using a weighted approach.
        
        This method implements a score combination strategy that:
        1. Normalizes scores from both methods to a comparable range
        2. Applies method-specific weights based on query characteristics
        3. Considers result positions in both rankings
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Dictionary mapping chunk IDs to combined scores
        """
        combined_scores = {}
        
        # Create position-based dictionaries
        dense_positions = {r.chunk_id: (i, r.score) for i, r in enumerate(dense_results)}
        sparse_positions = {r.chunk_id: (i, r.score) for i, r in enumerate(sparse_results)}
        
        # Get all unique chunk IDs
        all_chunks = set(dense_positions.keys()) | set(sparse_positions.keys())
        
        # Calculate score normalization factors
        dense_max = max((score for _, score in dense_positions.values()), default=1)
        sparse_max = max((score for _, score in sparse_positions.values()), default=1)
        
        for chunk_id in all_chunks:
            # Get positions and scores, defaulting to last position if not found
            dense_pos, dense_score = dense_positions.get(
                chunk_id, 
                (len(dense_results), 0)
            )
            sparse_pos, sparse_score = sparse_positions.get(
                chunk_id,
                (len(sparse_results), 0)
            )
            
            # Normalize scores to 0-1 range
            dense_norm = dense_score / dense_max if dense_max != 0 else 0
            sparse_norm = sparse_score / sparse_max if sparse_max != 0 else 0
            
            # Position-based weight adjustment
            position_weight = 1.0 / (1 + min(dense_pos, sparse_pos))
            
            # Combine scores with method weights and position adjustment
            combined_scores[str(chunk_id)] = (
                weights['dense'] * dense_norm +  # Dense retrieval weight
                weights['sparse'] * sparse_norm   # Sparse retrieval weight
            ) * position_weight
        
        return combined_scores
    
    def _calculate_result_metrics(
        self,
        result: SearchResult,
        query: str,
        current_query: str
    ) -> Dict[str, float]:
        """
        Calculate objective quality metrics for each search result. These metrics help us 
        evaluate results using measurable criteria alongside LLM confidence scores.
        """
        try:
            # Calculate semantic similarity with both original and current queries
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                device=self.device
            )
            current_query_embedding = self.embedding_model.encode(
                current_query,
                normalize_embeddings=True,
                device=self.device
            )
            text_embedding = self.embedding_model.encode(
                result.text,
                normalize_embeddings=True,
                device=self.device
            )
            
            # Calculate similarities
            original_sim = float(np.dot(text_embedding, query_embedding))
            current_sim = float(np.dot(text_embedding, current_query_embedding))
            
            # Calculate text quality metrics
            words = result.text.lower().split()
            unique_words = len(set(words)) / len(words) if words else 0
            
            # Calculate keyword matches
            query_words = set(query.lower().split())
            matches = sum(1 for word in query_words if word in result.text.lower())
            keyword_score = matches / len(query_words) if query_words else 0
            
            return {
                'semantic_relevance': (original_sim + current_sim) / 2,
                'text_quality': unique_words,
                'keyword_match': keyword_score,
                'has_context': 1.0 if result.context else 0.0
            }
        except Exception as e:
            logger.error(f"Error calculating result metrics: {str(e)}")
            return {
                'semantic_relevance': 0.0,
                'text_quality': 0.0,
                'keyword_match': 0.0,
                'has_context': 0.0
            }

    def _select_best_results(
        self,
        accumulated_results: Dict[int, Dict],
        result_metrics: Dict[int, Dict[str, float]],
        iterations: List[SearchIteration]
    ) -> List[SearchResult]:
        """
        Select best results using a comprehensive scoring approach that incorporates:
        1. Combined scores from dense and sparse retrieval
        2. Objective metrics like semantic similarity and text quality
        3. LLM confidence scores
        4. Result consistency across iterations
        """
        try:
            final_scores = {}
            
            for chunk_id, data in accumulated_results.items():
                metrics = result_metrics[chunk_id]
                
                # Get combined scores from iterations where this result appeared
                retrieval_scores = []
                llm_confidences = []
                
                for iteration in iterations:
                    if str(chunk_id) in iteration.combined_scores:
                        # Use the combined score from dense and sparse retrieval
                        retrieval_scores.append(iteration.combined_scores[str(chunk_id)])
                        
                    if iteration.reasoning != None and str(chunk_id) in iteration.reasoning.relevance_findings:
                        # Include LLM confidence scores
                        llm_confidences.append(iteration.reasoning.confidence_score)
                
                # Calculate average scores with defaults
                avg_retrieval_score = (
                    sum(retrieval_scores) / len(retrieval_scores)
                    if retrieval_scores else 0.5
                )
                avg_llm_confidence = (
                    sum(llm_confidences) / len(llm_confidences)
                    if llm_confidences else 0.5
                )
                
                # Calculate comprehensive final score
                objective_score = (
                    0.3 * avg_retrieval_score +           # Combined dense/sparse retrieval score
                    0.2 * metrics['semantic_relevance'] +  # Semantic similarity
                    0.2 * metrics['text_quality'] +        # Text quality
                    0.1 * metrics['keyword_match']         # Direct keyword matches
                )
                
                subjective_score = (
                    0.1 * avg_llm_confidence +            # LLM confidence
                    0.1 * (len(data['iterations_found']) / len(iterations))  # Consistency
                )
                
                # Store detailed score breakdown
                final_scores[chunk_id] = {
                    'total_score': objective_score + subjective_score,
                    'retrieval_score': avg_retrieval_score,
                    'objective_score': objective_score,
                    'subjective_score': subjective_score,
                    'llm_confidence': avg_llm_confidence
                }
            
            # Sort and prepare final results with score metadata
            sorted_results = sorted(
                accumulated_results.items(),
                key=lambda x: final_scores[x[0]]['total_score'],
                reverse=True
            )
            
            final_results = []
            for chunk_id, data in sorted_results[:self.results_per_step]:
                result = data['result']
                result.metadata['score_breakdown'] = final_scores[chunk_id]
                final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error selecting best results: {str(e)}")
            return list(accumulated_results.values())[:self.results_per_step]
    
    async def search(
    self,
    query: str,
    return_steps: bool = False
) -> Tuple[List[SearchResult], Optional[List[SearchIteration]]]:
        """
        Perform an enhanced iterative search that combines multiple retrieval strategies
        with careful result accumulation and intelligent scoring.
        
        This implementation addresses several key aspects:
        1. Properly handles final query refinements
        2. Uses both objective metrics and LLM insights for scoring
        3. Accumulates results across iterations for comprehensive coverage
        4. Maintains detailed tracking of search progression
        """
        try:
            # Initialize tracking variables for our search process
            current_query = query
            iterations = []
            accumulated_results = {}  # Stores all unique results we find
            result_metrics = {}      # Stores our objective measurements
            
            # First, perform our initial retrieval with the original query
            logger.info(f"Performing initial retrieval with query: {current_query}")
            query_analysis = self.query_classifier.analyze_query(current_query)
            
            current_query = query_analysis.corrected_query
            logger.info(f"Corrected query: {current_query}")
            logger.info(f"Query type: {query_analysis.query_type}")

            dense_results = await self._get_dense_results(
                current_query,
                self.results_per_step
            )
            sparse_results = self._get_sparse_results(
                current_query,
                self.results_per_step
            )
            
            current_results = self._merge_results(
                dense_results,
                sparse_results,
                query_analysis.weights,
                query_analysis.query_type,
                self.results_per_step
            )
            
            if not current_results:
                logger.warning(f"No results found for initial query: {current_query}")
                return [], None if not return_steps else None
                
            # Process and store our initial results
            logger.info(f"Processing initial results...")
            for result in current_results:
                logger.info(f"Processing result: {result}")
                if result.chunk_id not in result_metrics:
                    result_metrics[result.chunk_id] = self._calculate_result_metrics(
                        result=result,
                        query=query,
                        current_query=current_query
                    )
                    
                accumulated_results[result.chunk_id] = {
                    'result': result,
                    'metrics': result_metrics[result.chunk_id],
                    'iterations_found': [0]
                }

            # Now perform exactly max_iterations rounds of LLM analysis and retrieval
            for iteration in range(self.max_iterations):
                # Get LLM's analysis of current results
                reasoning = await self._get_reasoned_analysis(
                    current_query,
                    current_results,
                    iterations
                )
                
                # Record the completed iteration
                iterations.append(SearchIteration(
                    query=current_query,
                    results=current_results,
                    reasoning=reasoning,
                    combined_scores=self._combine_scores(dense_results, sparse_results, query_analysis.weights),
                    timestamp=time.time()
                ))
                
                # If LLM suggests a refinement, use it for the next retrieval
                if reasoning.suggested_refinement:
                    current_query = reasoning.suggested_refinement
                else:
                    # If no refinement suggested, use original query as fallback
                    current_query = query
                    
                # Perform the next retrieval with either the refined or original query
                logger.info(f"Performing retrieval {iteration + 2} with query: {current_query}")
                
                logger.info(f"Query type: {query_analysis.query_type}")

                dense_results = await self._get_dense_results(
                    current_query,
                    self.results_per_step
                )
                sparse_results = self._get_sparse_results(
                    current_query,
                    self.results_per_step
                )
                
                query_analysis = self.query_classifier.analyze_query(current_query)
                current_results = self._merge_results(
                    dense_results,
                    sparse_results,
                    query_analysis.weights,
                    query_analysis.query_type,
                    self.results_per_step
                )
                
                # Process and store results from this retrieval
                logger.info(f"Processing results for iteration {iteration + 2}...")
                for result in current_results:
                    logger.info(f"Processing result: {result}")
                    if result.chunk_id not in result_metrics:
                        result_metrics[result.chunk_id] = self._calculate_result_metrics(
                            result=result,
                            query=query,
                            current_query=current_query
                        )
                        
                    if result.chunk_id not in accumulated_results:
                        accumulated_results[result.chunk_id] = {
                            'result': result,
                            'metrics': result_metrics[result.chunk_id],
                            'iterations_found': [iteration + 1]
                        }
                    else:
                        accumulated_results[result.chunk_id]['iterations_found'].append(iteration + 1)
            
            # Record the last iteration
            iterations.append(SearchIteration(
                query=current_query,
                results=current_results,
                reasoning=None,
                combined_scores=self._combine_scores(dense_results, sparse_results, query_analysis.weights),
                timestamp=time.time()
            ))

            # Select our final results using our comprehensive scoring
            final_results = self._select_best_results(
                accumulated_results=accumulated_results,
                result_metrics=result_metrics,
                iterations=iterations
            )
            
            # Add reasoning to final results
            for result in final_results:
                relevant_iteration = next(
                    (it for it in reversed(iterations) 
                    if it.reasoning != None and  str(result.chunk_id) in it.reasoning.relevance_findings),
                    None
                )
                if relevant_iteration:
                    result.reasoning = relevant_iteration.reasoning.reasoning_explanation
            
            if return_steps:
                return final_results, iterations
            return final_results, None
            
        except Exception as e:
            logger.error(f"Error in search process: {str(e)}")
            logger.error(f"Initial query: {query}")
            return [], None if not return_steps else None
    
    def _apply_temporal_scoring(
        self,
        result: SearchResult,
        constraints: TemporalConstraints
    ) -> float:
        """Calculate temporal relevance score for a result"""
        doc_timestamp = datetime.fromisoformat(result.metadata.get('created_time', ''))
        now = datetime.now()
        
        # Handle different time frames
        if constraints.time_frame == TimeFrame.ALL_TIME:
            days_old = (now - doc_timestamp).days
            return 1.0 * (0.95 ** days_old)
            
        if constraints.time_frame == TimeFrame.STRICT:
            if (constraints.start_date and doc_timestamp < constraints.start_date or
                constraints.end_date and doc_timestamp > constraints.end_date):
                return 0.0
            return 1.0
            
        # Flexible scoring with decay
        if constraints.start_date and doc_timestamp < constraints.start_date:
            days_before = (constraints.start_date - doc_timestamp).days
            return 0.5 * (0.9 ** days_before)
            
        if constraints.end_date and doc_timestamp > constraints.end_date:
            days_after = (doc_timestamp - constraints.end_date).days
            return 0.5 * (0.9 ** days_after)
            
        # Document is within range
        time_range = (constraints.end_date or now) - (constraints.start_date or doc_timestamp)
        position = (doc_timestamp - (constraints.start_date or doc_timestamp)).total_seconds()
        relative_position = position / max(time_range.total_seconds(), 1)
        
        return 1.0 + (0.5 * relative_position)

    async def search_with_parameters(
        self,
        params: SearchParameters,
        return_steps: bool = False
    ) -> Tuple[List[SearchResult], Optional[List[SearchIteration]]]:
        """
        Perform search with combined UI and natural language temporal parameters

        This method implements a two-step temporal processing approach:
        1. First, it parses any temporal expressions in the natural language query
        2. Then, it combines these with any UI-specified temporal constraints
        """
        try:
            # First, parse temporal expressions from the query
            cleaned_query, nl_temporal = self.temporal_parser.parse_query(params.query)
            
            # Log the parsing results for transparency
            logger.info(
                f"Temporal parsing results:"
                f"\nOriginal query: {params.query}"
                f"\nCleaned query: {cleaned_query}"
                f"\nDetected temporal constraints: {nl_temporal}"
            )
            
            # Update search parameters with natural language temporal info
            params.nl_temporal = nl_temporal
            params.query = cleaned_query

            # Get effective temporal constraints
            constraints = params.get_effective_constraints()
            
            # Log search parameters
            logger.info(
                f"Final search configuration:"
                f"\nQuery: {params.query}"
                f"\nUI Temporal: {params.ui_temporal}"
                f"\nNL Temporal: {params.nl_temporal}"
                f"\nEffective Constraints: {constraints}"
            )
            
            # Perform base search
            results, steps = await self.search(params.query, return_steps)
            
            # Apply temporal filtering and scoring
            if constraints.has_constraints:
                filtered_results = []
                for result in results:
                    temporal_score = self._apply_temporal_scoring(result, constraints)
                    
                    if temporal_score > 0:
                        result.score *= temporal_score
                        result.metadata['temporal_score'] = temporal_score
                        filtered_results.append(result)
                
                # Sort by adjusted scores
                filtered_results.sort(key=lambda x: x.score, reverse=True)
                results = filtered_results
            
            return results, steps
            
        except Exception as e:
            logger.error(f"Error in temporal search: {str(e)}")
            return [], None if not return_steps else None
        
    async def search_with_feedback(
        self,
        query: str,
        relevance_feedback: Optional[Dict[int, float]] = None
    ) -> Tuple[List[SearchResult], List[SearchIteration]]:
        """
        Perform search with optional relevance feedback from previous results.
        
        This method enhances the search process by incorporating user feedback
        on result relevance to improve subsequent queries.
        
        Args:
            query: Search query
            relevance_feedback: Dictionary mapping chunk_ids to relevance scores
            
        Returns:
            Tuple of (results, search_iterations)
        """
        # TODO: Implement relevance feedback mechanism
        # For now, return regular search results
        return await self.search(query, return_steps=True)