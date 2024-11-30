def _extract_key_concepts(self, query: str) -> Set[str]:
    """
    Extract important concepts and terms from a query using NLP techniques.
    
    This function uses multiple approaches to identify key concepts:
    1. Named entity recognition
    2. Noun phrase extraction
    3. Important keyword identification
    """
    key_concepts = set()
    
    # Tokenize and tag parts of speech
    tokens = word_tokenize(query.lower())
    pos_tags = pos_tag(tokens)
    
    # Extract noun phrases using chunking
    grammar = """
        NP: {<DT>?<JJ>*<NN.*>+}     # Chunk determiners, adjectives, and nouns
        CP: {<JJR|JJS><IN><NN.*>+}  # Comparative phrases
    """
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    
    # Extract concepts from noun phrases
    for subtree in tree.subtrees(filter=lambda t: t.label() in {'NP', 'CP'}):
        concept = ' '.join(word for word, tag in subtree.leaves())
        if len(concept.split()) > 1:  # Only keep multi-word concepts
            key_concepts.add(concept)
    
    # Add single important terms (nouns, verbs, adjectives)
    important_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'JJ'}
    for word, tag in pos_tags:
        if tag[:2] in important_tags and len(word) > 3:
            key_concepts.add(word)
    
    return key_concepts

def _construct_enhanced_query(
    self,
    original_query: str,
    key_concepts: Set[str],
    max_concepts: int = 3
) -> str:
    """
    Construct an enhanced search query that combines the original query with key concepts.
    
    The function creates a more comprehensive query by:
    1. Keeping the original query intent
    2. Adding the most relevant discovered concepts
    3. Maintaining a natural language structure
    """
    # Remove concepts that are already in the original query
    original_lower = original_query.lower()
    new_concepts = {
        concept for concept in key_concepts
        if concept.lower() not in original_lower
    }
    
    # Score concepts by relevance to original query
    concept_scores = {}
    for concept in new_concepts:
        # Calculate semantic similarity using embeddings
        concept_embedding = self.embedding_model.encode([concept])[0]
        query_embedding = self.embedding_model.encode([original_query])[0]
        similarity = cosine_similarity(
            concept_embedding.reshape(1, -1),
            query_embedding.reshape(1, -1)
        )[0][0]
        
        # Score also considers concept specificity
        specificity = len(concept.split())  # Multi-word concepts are usually more specific
        concept_scores[concept] = similarity * (1 + 0.1 * specificity)
    
    # Select top concepts
    top_concepts = sorted(
        concept_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:max_concepts]
    
    # Construct enhanced query
    if original_query.lower().startswith(('what', 'who', 'where', 'when', 'why', 'how')):
        # For question queries, append concepts naturally
        enhanced_query = f"{original_query} considering {', '.join(c[0] for c in top_concepts)}"
    else:
        # For keyword queries, combine with AND logic
        enhanced_query = f"{original_query} AND ({' OR '.join(c[0] for c in top_concepts)})"
    
    return enhanced_query


def _calculate_information_density(self, text: str) -> float:
    """
    Calculate the information density of a text passage.
    
    Considers factors like:
    - Unique terms ratio
    - Named entity density
    - Technical term frequency
    - Sentence complexity
    """
    words = word_tokenize(text.lower())
    unique_ratio = len(set(words)) / len(words)
    
    # Named entity density
    doc = self.nlp(text)
    entity_density = len(doc.ents) / len(words)
    
    # Average sentence complexity (by length)
    sentences = sent_tokenize(text)
    avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences])
    sentence_complexity = min(avg_sentence_length / 20, 1.0)  # Normalize
    
    # Combine metrics
    return (0.4 * unique_ratio + 
            0.3 * entity_density + 
            0.3 * sentence_complexity)

def _apply_diversity_penalty(self, ranking_scores: Dict[int, Dict]):
    """
    Apply penalties to similar results to promote diversity in top results.
    
    Uses text similarity to identify and penalize redundant content.
    """
    for id1, score1 in ranking_scores.items():
        for id2, score2 in ranking_scores.items():
            if id1 < id2:  # Check each pair once
                similarity = self._calculate_text_similarity(
                    score1['result'].text,
                    score2['result'].text
                )
                if similarity > 0.7:  # High similarity threshold
                    # Apply penalty to lower-scored result
                    if score1['final_score'] > score2['final_score']:
                        ranking_scores[id2]['final_score'] *= (1 - similarity * 0.3)
                    else:
                        ranking_scores[id1]['final_score'] *= (1 - similarity * 0.3)

def _rerank_final_results(
    self,
    results: List[SearchResult],
    original_query: str,
    query_evolution: List[Dict]
) -> List[SearchResult]:
    """
    Rerank results considering the complete search context and query evolution.
    
    This function implements a sophisticated reranking system that considers:
    1. Original result scores
    2. Query evolution history
    3. Content diversity
    4. Information density
    """
    # Calculate initial ranking scores
    ranking_scores = {}
    
    for result in results:
        # Start with normalized base score
        base_score = result.score / max(r.score for r in results)
        
        # Calculate query relevance across evolution
        query_relevance = 0.0
        for query_info in query_evolution:
            # Semantic similarity to each query version
            text_embedding = self.embedding_model.encode([result.text])[0]
            query_embedding = self.embedding_model.encode([query_info['query']])[0]
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                query_embedding.reshape(1, -1)
            )[0][0]
            
            # Weight by query confidence
            query_relevance += similarity * query_info['confidence']
        
        # Normalize query relevance
        query_relevance = query_relevance / len(query_evolution)
        
        # Calculate information density
        info_density = self._calculate_information_density(result.text)
        
        # Combine factors into final score
        ranking_scores[result.chunk_id] = {
            'final_score': 0.4 * base_score + 
                         0.4 * query_relevance + 
                         0.2 * info_density,
            'result': result
        }
    
    # Apply diversity penalty
    self._apply_diversity_penalty(ranking_scores)
    
    # Sort by final score and return results
    ranked_results = sorted(
        ranking_scores.values(),
        key=lambda x: x['final_score'],
        reverse=True
    )
    
    return [item['result'] for item in ranked_results]


async def search(
    self,
    query: str,
    return_steps: bool = False
) -> Tuple[List[SearchResult], Optional[List[SearchIteration]]]:
    """
    Enhanced iterative chain-of-thought guided search with comprehensive final retrieval.
    
    This implementation adds a final search phase that:
    1. Combines insights from all iterations
    2. Performs a final broad search using accumulated knowledge
    3. Re-ranks results considering the complete search context
    """
    try:
        current_query = query
        iterations = []
        accumulated_results = {}  # Track all unique results
        query_evolution = []     # Track query refinements
        
        # Initial search iterations
        for iteration in range(self.max_iterations):
            # Analyze query
            query_analysis = self.query_classifier.analyze_query(current_query)
            logger.info(f"Query type: {query_analysis.query_type.value}, "
                     f"Confidence: {query_analysis.confidence:.2f}")
            
            # Get and merge results
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
            
            # Store unique results with their scores
            for result in current_results:
                if result.chunk_id not in accumulated_results:
                    accumulated_results[result.chunk_id] = result
                else:
                    # Keep the highest score if we've seen this result before
                    accumulated_results[result.chunk_id].score = max(
                        accumulated_results[result.chunk_id].score,
                        result.score
                    )
            
            # Get reasoned analysis
            reasoning = await self._get_reasoned_analysis(
                current_query,
                current_results,
                iterations
            )
            
            # Record iteration
            iterations.append(SearchIteration(
                query=current_query,
                results=current_results,
                reasoning=reasoning,
                combined_scores=self._combine_scores(dense_results, sparse_results),
                timestamp=time.time()
            ))
            
            # Track query evolution
            query_evolution.append({
                'query': current_query,
                'confidence': reasoning.confidence_score,
                'key_findings': reasoning.key_findings
            })
            
            # Check termination conditions
            if (not reasoning.suggested_refinement or 
                reasoning.confidence_score >= self.min_confidence_threshold):
                break
                
            current_query = reasoning.suggested_refinement
        
        # Perform final comprehensive search
        final_results = await self._perform_final_search(
            original_query=query,
            query_evolution=query_evolution,
            accumulated_results=accumulated_results,
            iterations=iterations
        )
        
        # Add reasoning to final results
        for result in final_results:
            relevant_iteration = next(
                (it for it in reversed(iterations) 
                 if str(result.chunk_id) in it.reasoning.relevance_findings),
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

async def _perform_final_search(
    self,
    original_query: str,
    query_evolution: List[Dict],
    accumulated_results: Dict[int, SearchResult],
    iterations: List[SearchIteration]
) -> List[SearchResult]:
    """
    Perform a final comprehensive search using insights from all iterations.
    """
    # Analyze query evolution to identify key concepts
    key_concepts = set()
    for query_info in query_evolution:
        key_concepts.update(self._extract_key_concepts(query_info['query']))
        key_concepts.update(query_info['key_findings'])
    
    # Construct an enhanced final query
    enhanced_query = self._construct_enhanced_query(
        original_query,
        key_concepts
    )
    
    # Perform final retrieval with enhanced query
    final_dense_results = await self._get_dense_results(
        enhanced_query,
        self.results_per_step * 2  # Broader search
    )
    final_sparse_results = self._get_sparse_results(
        enhanced_query,
        self.results_per_step * 2
    )
    
    # Merge new results with accumulated results
    all_results = list(accumulated_results.values()) + final_dense_results + final_sparse_results
    
    # Re-rank considering full context
    return self._rerank_final_results(
        all_results,
        original_query=original_query,
        query_evolution=query_evolution
    )