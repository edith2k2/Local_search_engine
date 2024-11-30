import asyncio
import os
from pathlib import Path
import logging
from anthropic import AsyncAnthropic
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from preprocessing import AsyncDocumentProcessor, BatchConfig
from retriever import ChainOfThoughtRetriever, SearchResult

# Set up logging to track our test execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_all_retriever_functions():
    """
    Test each function in the ChainOfThoughtRetriever class to demonstrate its functionality.
    We'll create a minimal test environment and methodically test each component.
    """
    try:
        # First, let's set up our test environment
        logger.info("Setting up test environment...")
        
        # Initialize necessary components
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
        
        # Create a small test document
        test_doc_content = """
        Artificial Intelligence Overview
        
        AI is a broad field of computer science focused on creating intelligent machines.
        Machine learning is a subset of AI that uses data to improve performance.
        Deep learning is a type of machine learning using neural networks.
        """
        
        # Create test directory and document
        test_dir = Path("test_documents")
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test_article.txt"
        with open(test_file, "w") as f:
            f.write(test_doc_content)
        
        # Initialize processor for document processing
        processor = AsyncDocumentProcessor(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2",
            anthropic_api_key=api_key,
            device='cpu',  # Using CPU for testing
            batch_config=BatchConfig(
                embeddings=32,
                context=10,
                faiss=1000,
                documents=5,
                process=4
            )
        )
        
        # Process test document
        logger.info("Processing test document...")
        processed_docs = await processor.process_documents([str(test_file)])
        
        # Initialize the retriever
        logger.info("Initializing retriever...")
        retriever = ChainOfThoughtRetriever(
            documents=processed_docs,
            embedding_model=processor.embedding_model,
            anthropic_client=processor.client,
            device='cpu'
        )
        
        # Now let's test each function in the retriever
        
        # 1. Test _initialize_indices
        logger.info("\nTesting _initialize_indices...")
        # This was called during initialization, let's verify the structures
        assert len(retriever.all_chunks) > 0, "Chunks were not initialized"
        assert len(retriever.doc_indices) > 0, "Document indices were not initialized"
        assert len(retriever.bm25_indices) > 0, "BM25 indices were not initialized"
        logger.info("Index initialization successful")

        # 2. Test _get_dense_results
        logger.info("\nTesting _get_dense_results...")
        dense_results = await retriever._get_dense_results(
            query="What is artificial intelligence?",
            k=3
        )
        logger.info(f"Found {len(dense_results)} dense results")
        
        # 3. Test _get_sparse_results
        logger.info("\nTesting _get_sparse_results...")
        sparse_results = retriever._get_sparse_results(
            query="What is machine learning?",
            k=3
        )
        logger.info(f"Found {len(sparse_results)} sparse results")
        
        # 4. Test _merge_results
        logger.info("\nTesting _merge_results...")
        merged_results = retriever._merge_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=3
        )
        logger.info(f"Merged into {len(merged_results)} results")
        
        # 5. Test _check_redundancy
        logger.info("\nTesting _check_redundancy...")
        redundant_pairs = retriever._check_redundancy(merged_results)
        logger.info(f"Found {len(redundant_pairs)} redundant pairs")
        
        # 6. Test _get_reasoned_analysis
        logger.info("\nTesting _get_reasoned_analysis...")
        reasoning_step = await retriever._get_reasoned_analysis(
            query="What is AI?",
            results=merged_results,
            previous_steps=[]
        )
        logger.info(f"Generated reasoning with confidence: {reasoning_step.confidence_score}")
        
        # 7. Test _combine_scores
        logger.info("\nTesting _combine_scores...")
        combined_scores = retriever._combine_scores(
            dense_results=dense_results,
            sparse_results=sparse_results
        )
        logger.info(f"Combined scores for {len(combined_scores)} results")
        
        # 8. Test main search function
        logger.info("\nTesting main search function...")
        results, steps = await retriever.search(
            query="What is artificial intelligence?",
            return_steps=True
        )
        logger.info(f"Search completed with {len(results)} results and {len(steps)} steps")
        
        # 9. Test search_with_feedback
        logger.info("\nTesting search_with_feedback...")
        feedback_results, feedback_steps = await retriever.search_with_feedback(
            query="What is machine learning?",
            relevance_feedback={0: 1.0, 1: 0.5}  # Sample feedback scores
        )
        logger.info("Search with feedback completed")
        
        # Clean up test files
        logger.info("\nCleaning up test files...")
        test_file.unlink()
        test_dir.rmdir()
        
        logger.info("\nAll function tests completed successfully!")
        
        # Return some sample results for inspection
        return {
            'dense_results': dense_results,
            'sparse_results': sparse_results,
            'merged_results': merged_results,
            'reasoning': reasoning_step,
            'search_results': results,
            'search_steps': steps
        }
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise

async def main():
    """Main execution function that runs our tests and displays results"""
    try:
        logger.info("Starting retriever function tests...")
        test_results = await test_all_retriever_functions()
        
        # Display some sample results
        logger.info("\n=== Sample Results ===")
        
        if test_results['search_results']:
            logger.info("\nFirst search result:")
            result = test_results['search_results'][0]
            logger.info(f"Source: {Path(result.source).name}")
            logger.info(f"Score: {result.score}")
            logger.info(f"Text: {result.text[:100]}...")
        
        if test_results['search_steps']:
            logger.info("\nFirst reasoning step:")
            step = test_results['search_steps'][0]
            logger.info(f"Query: {step.query}")
            logger.info(f"Confidence: {step.reasoning.confidence_score}")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())