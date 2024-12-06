import re
from typing import List, Set
import spacy
from pathlib import Path
import logging

class Tokenizer:
    """
    A high-performance tokenizer optimized for search engine applications.
    Focuses on accuracy and speed using spaCy's efficient processing.
    """
    
    def __init__(self):
        # Load the small English model for efficiency
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        # Optimize spaCy pipeline - disable components we don't need
        self.nlp.disable_pipes(["parser", "ner"])
        
        # Compile regex patterns once
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Technical terms that should always be preserved
        self.special_terms = {
            'api', 'json', 'http', 'sql', 'html', 'css', 'js', 'url',
            'aws', 'cli', 'git', 'ssh', 'ftp', 'xml', 'ssl', 'tls',
            'cpu', 'gpu', 'ram', 'ip', 'dns', 'ai', 'ml'
        }
        
        # Load batch_data for faster processing
        self.nlp.batch_size = 1000
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text efficiently using spaCy with custom rules.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of cleaned tokens
        """
        # Clean whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
                
            # Get lowercase version for checking
            lower_token = token.text.lower()
            
            # Preserve special terms
            if lower_token in self.special_terms:
                tokens.append(lower_token)
                continue
            
            # Skip very short tokens
            if len(token.text) < 2:
                continue
                
            # Get lemma for regular words
            tokens.append(token.lemma_.lower())
            
        return tokens
    
    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize multiple texts efficiently using spaCy's pipe.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            List of token lists
        """
        results = []
        # Use spaCy's pipe for efficient batch processing
        for doc in self.nlp.pipe(texts):
            tokens = []
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                    
                lower_token = token.text.lower()
                if lower_token in self.special_terms:
                    tokens.append(lower_token)
                    continue
                
                if len(token.text) < 2:
                    continue
                    
                tokens.append(token.lemma_.lower())
            
            results.append(tokens)
            
        return results