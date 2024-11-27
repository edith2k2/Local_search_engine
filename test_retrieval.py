import asyncio
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from anthropic import AsyncAnthropic
import json
import tempfile
import shutil
import aiofiles
from dataclasses import asdict
from retriever import DocumentRetriever, SearchResult, SearchStep
from preprocessing import DocumentProcessor

async def setup_test_data():
    """Create sample documents and indices for testing."""
    # Sample document content
    processor = DocumentProcessor()
    output_dir = "output"
    
    # Load indices from disk
    await processor.load_indices(output_dir)
    documents = {
        "doc1.txt": {
            "metadata": {
                "file_path": "doc1.txt",
                "file_name": "doc1.txt",
                "file_type": ".txt",
                "created_time": datetime.now().isoformat(),
                "modified_time": datetime.now().isoformat(),
                "size_bytes": 1000,
                "num_chunks": 2
            },
            "chunks": [
                {
                    "chunk_id": 0,
                    "text": "Python is a popular programming language.",
                    "start_idx": 0,
                    "end_idx": 42,
                    "context": "Introduction to programming languages",
                    "embedding": np.random.rand(384)  # Example dimension
                },
                {
                    "chunk_id": 1,
                    "text": "Machine learning is transforming technology.",
                    "start_idx": 43,
                    "end_idx": 85,
                    "context": "Overview of AI applications",
                    "embedding": np.random.rand(384)
                }
            ],
            "bm25_index": None  # Will be created during test
        }
    }
    
    return documents

async def test_hybrid_search():
    """Test the hybrid search functionality combining BM25 and embeddings."""
    print("\nTesting hybrid search...")
    
    try:
        # Setup
        documents = await setup_test_data()
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        retriever = DocumentRetriever(documents, client)
        
        # Test query
        query = "Python programming"
        results = await retriever.hybrid_search(query)
        
        # Verify results
        assert len(results) > 0, "Search should return results"
        assert isinstance(results[0], SearchResult), "Results should be SearchResult objects"
        assert results[0].text is not None, "Results should contain text"
        print("✓ Hybrid search test passed")
        
    except Exception as e:
        print(f"✗ Hybrid search test failed: {str(e)}")
        raise

async def test_iterative_search():
    """Test the iterative LLM-guided search process."""
    print("\nTesting iterative search...")
    
    try:
        # Setup
        documents = await setup_test_data()
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        retriever = DocumentRetriever(documents, client)
        
        # Test complex query requiring multiple steps
        query = "Explain how Python is used in machine learning applications"
        results, search_history = await retriever.iterative_search(query)
        
        # Verify results and search process
        assert len(results) > 0, "Search should return results"
        assert len(search_history) > 0, "Search should record steps"
        assert isinstance(search_history[0], SearchStep), "History should contain SearchStep objects"
        assert search_history[0].reasoning is not None, "Steps should include LLM reasoning"
        print("✓ Iterative search test passed")
        
    except Exception as e:
        print(f"✗ Iterative search test failed: {str(e)}")
        raise

async def test_rank_fusion():
    """Test the rank fusion algorithm for combining search results."""
    print("\nTesting rank fusion...")
    
    try:
        # Setup
        documents = await setup_test_data()
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        retriever = DocumentRetriever(documents, client)
        
        # Create sample results
        bm25_results = [
            SearchResult(
                chunk_id=0,
                file_path="doc1.txt",
                text="Sample text 1",
                context="Sample context",
                score=0.8,
                retrieval_method="bm25"
            ),
            SearchResult(
                chunk_id=1,
                file_path="doc1.txt",
                text="Sample text 2",
                context="Sample context",
                score=0.6,
                retrieval_method="bm25"
            )
        ]
        
        embedding_results = [
            SearchResult(
                chunk_id=0,
                file_path="doc1.txt",
                text="Sample text 1",
                context="Sample context",
                score=0.9,
                retrieval_method="embedding"
            ),
            SearchResult(
                chunk_id=2,
                file_path="doc1.txt",
                text="Sample text 3",
                context="Sample context",
                score=0.7,
                retrieval_method="embedding"
            )
        ]
        
        # Test rank fusion
        combined_results = retriever.rank_fusion(bm25_results + embedding_results)
        
        # Verify results
        assert len(combined_results) > 0, "Rank fusion should return results"
        assert isinstance(combined_results[0], SearchResult), "Results should be SearchResult objects"
        assert combined_results[0].score >= combined_results[-1].score, "Results should be properly ranked"
        print("✓ Rank fusion test passed")
        
    except Exception as e:
        print(f"✗ Rank fusion test failed: {str(e)}")
        raise

async def test_embedding_computation():
    """Test the embedding computation functionality."""
    print("\nTesting embedding computation...")
    
    try:
        # Setup
        documents = await setup_test_data()
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        retriever = DocumentRetriever(documents, client)
        
        # Test texts
        texts = ["Sample query 1", "Sample query 2"]
        embeddings = await retriever.process_embeddings(texts)
        
        # Verify embeddings
        assert len(embeddings) == len(texts), "Should generate embedding for each text"
        assert isinstance(embeddings[0], np.ndarray), "Embeddings should be numpy arrays"
        assert embeddings[0].shape[-1] == 768, "Embedding dimension should match model"
        print("✓ Embedding computation test passed")
        
    except Exception as e:
        print(f"✗ Embedding computation test failed: {str(e)}")
        raise

async def test_search_persistence():
    """Test saving and loading search indices and results."""
    print("\nTesting search persistence...")
    
    try:
        # Setup temporary directory
        temp_dir = tempfile.mkdtemp()
        documents = await setup_test_data()
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        retriever = DocumentRetriever(documents, client)
        
        # Test query and save results
        query = "Python programming"
        results = await retriever.hybrid_search(query)
        
        # Save search state
        save_path = Path(temp_dir) / "search_state"
        save_path.mkdir(exist_ok=True)
        
        # Save results
        async with aiofiles.open(save_path / "results.json", "w") as f:
            await f.write(json.dumps([asdict(r) for r in results]))
        
        # Load and verify
        async with aiofiles.open(save_path / "results.json", "r") as f:
            loaded_results = json.loads(await f.read())
        
        assert len(loaded_results) == len(results), "Should maintain all results"
        print("✓ Search persistence test passed")
        
    except Exception as e:
        print(f"✗ Search persistence test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

async def run_all_tests():
    """Run all search engine tests."""
    print("Starting search engine test suite...")
    
    try:
        await test_hybrid_search()
        await test_iterative_search()
        await test_rank_fusion()
        await test_embedding_computation()
        await test_search_persistence()
        
        print("\n✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run tests
    asyncio.run(run_all_tests())