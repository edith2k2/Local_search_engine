from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from anthropic import AsyncAnthropic
from retriever import SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Represents a specific piece of text cited in an answer"""
    text: str
    source: str
    score: float
    context: Optional[str] = None

@dataclass
class GeneratedAnswer:
    """Contains the generated answer along with supporting information"""
    answer: str
    citations: List[Citation]
    confidence_score: float
    metadata: Dict[str, Any]

class AnswerGenerator:
    """
    Generates comprehensive answers from search results using an LLM.
    This class handles the transformation of search results into coherent
    answers with proper citations and confidence scoring.
    """
    
    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        model: str = "claude-3-sonnet-20240229",
        max_context_length: int = 5,
        temperature: float = 0.3
    ):
        """
        Initialize the answer generator with the specified configuration.
        
        Args:
            anthropic_client: Initialized AsyncAnthropic client
            model: The Claude model to use for answer generation
            max_context_length: Maximum number of search results to include in context
            temperature: Temperature setting for answer generation
        """
        self.client = anthropic_client
        self.model = model
        self.max_context_length = max_context_length
        self.temperature = temperature

    def _prepare_context(self, results: List[SearchResult]) -> str:
        """
        Prepare search results as context for the LLM prompt.
        
        Args:
            results: List of search results to format
            
        Returns:
            Formatted context string with source information
        """
        context_parts = []
        
        for i, result in enumerate(results[:self.max_context_length], 1):
            # Extract filename from path for cleaner citation
            source_name = Path(result.source).name
            
            # Format the passage with its source information
            passage = f"""Source {i}: {source_name}
            Relevance Score: {result.score:.2f}
            
            {result.text}
            
            Context: {result.context if result.context else 'No additional context available'}
            ---"""
            
            context_parts.append(passage)
        
        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a comprehensive prompt for answer generation.
        
        Args:
            query: The user's search query
            context: Formatted context from search results
            
        Returns:
            Complete prompt for the LLM
        """
        return f"""Generate a comprehensive answer to the following question using only the provided source passages. 

Question: {query}

Source Passages:

{context}

Instructions:
1. Synthesize information from the provided sources to create a clear, coherent answer
2. Use objective language and maintain accuracy to the source material
3. Include specific information from the sources with clear attribution
4. If the sources don't contain enough information to fully answer the query, acknowledge this
5. Structure the answer to build understanding progressively

Format the answer as follows:
- Start with a direct answer to the question
- Follow with supporting details and explanations
- End with a summary of key points
- Use "According to [Source X]" or similar phrases to attribute information

Answer:"""

    def _extract_citations(
        self,
        results: List[SearchResult],
        answer_text: str
    ) -> List[Citation]:
        """
        Create citations from search results that were most likely used in the answer.
        
        Args:
            results: Original search results
            answer_text: Generated answer text
            
        Returns:
            List of relevant citations
        """
        citations = []
        
        # Use top results that have content reflected in the answer
        for result in results[:self.max_context_length]:
            # Simple relevance check - could be enhanced with more sophisticated matching
            key_phrases = [phrase.strip() for phrase in result.text.split('.') if len(phrase.strip()) > 20]
            
            for phrase in key_phrases:
                # If any significant phrase from the result appears in the answer
                if phrase.lower() in answer_text.lower():
                    citations.append(Citation(
                        text=result.text,
                        source=result.source,
                        score=result.score,
                        context=result.context
                    ))
                    break  # One citation per result maximum
        
        return citations

    def _calculate_confidence(
        self,
        results: List[SearchResult],
        citations: List[Citation]
    ) -> float:
        """
        Calculate a confidence score for the generated answer.
        
        Args:
            results: Original search results
            citations: Extracted citations
            
        Returns:
            Confidence score between 0 and 1
        """
        if not results:
            return 0.0
        
        factors = {
            'citation_coverage': len(citations) / min(len(results), self.max_context_length),
            'result_scores': sum(r.score for r in results[:self.max_context_length]) / self.max_context_length if results else 0,
            'citation_scores': sum(c.score for c in citations) / len(citations) if citations else 0
        }
        
        # Weighted average of factors
        weights = {'citation_coverage': 0.4, 'result_scores': 0.3, 'citation_scores': 0.3}
        confidence = sum(score * weights[factor] for factor, score in factors.items())
        
        return min(max(confidence, 0.0), 1.0)

    async def generate_answer(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Optional[GeneratedAnswer]:
        """
        Generate a comprehensive answer from search results.
        
        Args:
            query: User's search query
            results: Search results to use for answer generation
            
        Returns:
            GeneratedAnswer object containing the answer and supporting information
        """
        try:
            if not results:
                logger.warning("No search results provided for answer generation")
                return None
            
            # Prepare context from search results
            context = self._prepare_context(results)
            
            # Generate answer using Claude
            prompt = self._create_prompt(query, context)
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer_text = response.content[0].text.strip()
            
            # Extract citations and calculate confidence
            citations = self._extract_citations(results, answer_text)
            confidence = self._calculate_confidence(results, citations)
            
            # Create metadata about the generation process
            metadata = {
                'model_used': self.model,
                'temperature': self.temperature,
                'context_length': len(results[:self.max_context_length]),
                'generation_time': response.usage.total_time_millis / 1000 if hasattr(response, 'usage') else None
            }
            
            return GeneratedAnswer(
                answer=answer_text,
                citations=citations,
                confidence_score=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return None