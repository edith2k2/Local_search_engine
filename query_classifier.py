from enum import Enum
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Defines different types of queries that might require different retrieval strategies"""
    FACTUAL = "factual"         # Simple fact-finding queries
    REASONING = "reasoning"      # Queries requiring analysis or explanation
    COMPARISON = "comparison"    # Queries about comparing or contrasting
    EXPLORATORY = "exploratory" # Open-ended, discovery-oriented queries
    PROCEDURAL = "procedural"   # How-to queries about processes or steps

@dataclass
class QueryAnalysis:
    """Stores the analysis results for a query"""
    query_type: QueryType
    weights: Dict[str, float]
    confidence: float = 1.0  # How confident we are in this classification

class QueryClassifier:
    """Analyzes queries to determine their type and appropriate retrieval weights"""
    
    def __init__(self):
        # Define characteristics and weights for each query type
        self.query_patterns = {
            QueryType.FACTUAL: {
                'keywords': ['what is', 'who is', 'when did', 'where is', 'which'],
                'weights': {'dense': 0.4, 'sparse': 0.6}  # Favor keyword matching
            },
            QueryType.REASONING: {
                'keywords': ['why', 'how does', 'explain', 'what are the reasons', 'analyze'],
                'weights': {'dense': 0.7, 'sparse': 0.3}  # Favor semantic understanding
            },
            QueryType.COMPARISON: {
                'keywords': ['compare', 'difference between', 'versus', 'vs', 'better', 'advantages'],
                'weights': {'dense': 0.6, 'sparse': 0.4}  # Balance both approaches
            },
            QueryType.EXPLORATORY: {
                'keywords': ['tell me about', 'describe', 'elaborate', 'what are', 'overview'],
                'weights': {'dense': 0.8, 'sparse': 0.2}  # Strongly favor semantic search
            },
            QueryType.PROCEDURAL: {
                'keywords': ['how to', 'steps', 'guide', 'process of', 'procedure'],
                'weights': {'dense': 0.5, 'sparse': 0.5}  # Equal weighting
            }
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyzes a query to determine its type and appropriate retrieval weights.
        
        Args:
            query: The search query string
            
        Returns:
            QueryAnalysis containing the query type and weights to use
        """
        query = query.lower().strip()
        
        # Check each query type for matches
        matches = []
        for query_type, pattern_info in self.query_patterns.items():
            # Count how many keywords match
            match_count = sum(1 for keyword in pattern_info['keywords'] 
                            if keyword in query)
            if match_count > 0:
                matches.append((query_type, match_count))
        
        if matches:
            # Sort by number of matches and take the best match
            best_match = max(matches, key=lambda x: x[1])
            query_type = best_match[0]
            confidence = min(best_match[1] / len(query.split()), 1.0)
        else:
            # Default to exploratory if no clear matches
            query_type = QueryType.EXPLORATORY
            confidence = 0.5
        
        return QueryAnalysis(
            query_type=query_type,
            weights=self.query_patterns[query_type]['weights'],
            confidence=confidence
        )