from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import spacy
import logging
from enum import Enum
from symspellpy import SymSpell, Verbosity
import pkg_resources
import re

logger = logging.getLogger(__name__)

class QueryType(Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    COMPARISON = "comparison"
    EXPLORATORY = "exploratory"
    PROCEDURAL = "procedural"

@dataclass
class QueryAnalysis:
    query_type: QueryType
    weights: Dict[str, float]
    confidence: float = 1.0
    features: Dict[str, float] = None
    corrections: Dict[str, str] = None
    original_query: str = None
    corrected_query: str = None

class QueryClassifier:
    """
    Query classifier with sophisticated spell correction using SymSpell algorithm.
    SymSpell offers several advantages:
    1. Very fast (about 1 million times faster than Levenshtein)
    2. More accurate for compound misspellings
    3. Considers word frequency for better suggestions
    """
    
    def __init__(self):
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize SymSpell
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Load dictionary with word frequencies
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        
        # Load bigram dictionary for compound corrections
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
            
        # Load dictionaries
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
        
        # Add domain-specific terms and their common misspellings
        # self.add_domain_vocabulary()
        
        # Intent patterns remain the same as before
        self.intent_patterns = {
            QueryType.FACTUAL: {
                'question_words': ['what', 'when', 'where', 'who', 'which'],
                'verbs': ['is', 'are', 'was', 'were', 'does'],
                'patterns': ['define', 'meaning of', 'definition of']
            },
            QueryType.REASONING: {
                'question_words': ['why', 'how'],
                'verbs': ['explain', 'causes', 'affects', 'influences', 'works'],
                'patterns': ['reason for', 'because', 'explain', 'understand']
            },
            QueryType.COMPARISON: {
                'markers': ['compare', 'versus', 'vs', 'difference', 'better', 'worse'],
                'patterns': ['compared to', 'differences between', 'pros and cons']
            },
            QueryType.EXPLORATORY: {
                'verbs': ['tell', 'describe', 'elaborate', 'discuss'],
                'patterns': ['tell me about', 'what are', 'information about', 'learn about']
            },
            QueryType.PROCEDURAL: {
                'markers': ['how to', 'steps', 'guide', 'tutorial', 'instructions'],
                'verbs': ['make', 'create', 'build', 'implement', 'setup', 'configure']
            }
        }
        
        self.retrieval_weights = {
            QueryType.FACTUAL: {'dense': 0.2, 'sparse': 0.8},
            QueryType.REASONING: {'dense': 0.75, 'sparse': 0.25},
            QueryType.COMPARISON: {'dense': 0.60, 'sparse': 0.40},
            QueryType.EXPLORATORY: {'dense': 0.80, 'sparse': 0.20},
            QueryType.PROCEDURAL: {'dense': 0.55, 'sparse': 0.45}
        }

    # def add_domain_vocabulary(self):
    #     """
    #     Add domain-specific terms and their common misspellings to the dictionary.
    #     This helps handle technical terms that might not be in the standard dictionary.
    #     """
    #     domain_terms = {
    #         # Technical terms
    #         'algorithm': ['algoritm', 'algorithim', 'algorythm'],
    #         'neural': ['nueral', 'nural', 'neuronal'],
    #         'network': ['netwok', 'netwerk', 'nework'],
    #         'machine': ['machin', 'machene', 'machien'],
    #         'learning': ['lerning', 'learnin', 'learing'],
    #         'training': ['traning', 'trainning', 'trianing'],
    #         'dataset': ['datatset', 'dateset', 'dataset'],
    #         'parameter': ['paramter', 'parameter', 'parametre'],
            
    #         # Query-specific terms
    #         'compare': ['compair', 'compar', 'compre'],
    #         'difference': ['diferent', 'diffrence', 'diferrence'],
    #         'explain': ['explian', 'explan', 'expain'],
    #         'implement': ['implment', 'implemnt', 'implementt']
    #     }
        
    #     # Add terms and their misspellings to the dictionary
    #     for correct, misspellings in domain_terms.items():
    #         # Add correct term with high frequency
    #         self.sym_spell.create_dictionary_entry(correct, 10000)
            
    #         # Add misspellings with lower frequency
    #         for misspelling in misspellings:
    #             self.sym_spell.create_dictionary_entry(misspelling, 1)

    def correct_query(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Correct spelling errors in the query using SymSpell.
        Returns both the corrected query and a dictionary of corrections made.
        """
        # First, try compound correction for the whole query
        suggestions = self.sym_spell.lookup_compound(
            query,
            max_edit_distance=2,
            transfer_casing=True  # Preserve original casing
        )
        
        if not suggestions:
            return query, {}
            
        best_suggestion = suggestions[0]
        corrected_query = best_suggestion.term
        
        # Track individual word corrections
        corrections = {}
        original_words = query.split()
        corrected_words = corrected_query.split()
        
        # Match original words with corrections
        for orig, corr in zip(original_words, corrected_words):
            if orig.lower() != corr.lower():
                corrections[orig] = corr
        
        return corrected_query, corrections

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query with advanced spell correction and classification.
        """
        try:
            # First, correct any spelling errors
            corrected_query, corrections = self.correct_query(query)
            
            # Process with spaCy
            doc = self.nlp(corrected_query)
            
            # Calculate type scores
            type_scores = self._calculate_type_scores(doc, corrected_query)
            
            # Get predicted type and confidence
            predicted_type = max(type_scores.items(), key=lambda x: x[1])
            query_type = predicted_type[0]
            confidence = predicted_type[1]
            
            # Adjust weights based on confidence
            weights = self.retrieval_weights[query_type].copy()
            if confidence < 0.5:
                for key in weights:
                    weights[key] = 0.5 + (weights[key] - 0.5) * confidence
            
            return QueryAnalysis(
                query_type=query_type,
                weights=weights,
                confidence=confidence,
                features=type_scores,
                corrections=corrections if corrections else None,
                original_query=query,
                corrected_query=corrected_query
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return QueryAnalysis(
                query_type=QueryType.EXPLORATORY,
                weights={'dense': 0.5, 'sparse': 0.5},
                confidence=0.0,
                original_query=query
            )

    def _calculate_type_scores(self, doc, query: str) -> Dict[QueryType, float]:
        """Calculate scores for each query type using linguistic features."""
        scores = {qt: 0.0 for qt in QueryType}
        query_lower = query.lower()
        
        for query_type, patterns in self.intent_patterns.items():
            score = 0.0
            
            # Check individual words and patterns
            for key in ['question_words', 'verbs', 'markers']:
                if key in patterns:
                    score += sum(word in query_lower.split() 
                               for word in patterns[key]) * 0.3
            
            # Check multi-word patterns
            if 'patterns' in patterns:
                score += sum(pattern in query_lower 
                           for pattern in patterns['patterns']) * 0.5
            
            # Add linguistic feature scores
            if query_type == QueryType.FACTUAL and any(token.tag_ in ['WDT', 'WP', 'WRB'] 
                                                      for token in doc):
                score += 0.4
            elif query_type == QueryType.REASONING and any(token.text.lower() == 'why' 
                                                         for token in doc):
                score += 0.6
            elif query_type == QueryType.COMPARISON and any(token.dep_ == 'amod' 
                                                          for token in doc):
                score += 0.4
            elif query_type == QueryType.PROCEDURAL and doc[0].pos_ == 'VERB':
                score += 0.4
            
            scores[query_type] = min(score, 1.0)
        
        return scores