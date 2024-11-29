import asyncio
from sentence_transformers import SentenceTransformer
from anthropic import AsyncAnthropic
import torch
from retriever import ChainOfThoughtRetriever
from preprocessing import AsyncDocumentProcessor
import os
from pathlib import Path
import numpy as np

async def initialize_search_system(processed_documents, api_key):
    # Set up the embedding model
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # Determine the best available device
    if torch.cuda.is_available():
        device = 'cuda'
        embedding_model.to('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        embedding_model.to('mps')
    else:
        device = 'cpu'
        embedding_model.to('cpu')
    
    # Initialize the Anthropic client
    anthropic_client = AsyncAnthropic(api_key=api_key)
    
    # Create the retriever
    retriever = ChainOfThoughtRetriever(
        documents=processed_documents,
        embedding_model=embedding_model,
        anthropic_client=anthropic_client,
        device=device,  # Pass the device explicitly
        max_steps=3,
        results_per_step=5
    )
        # In your main code, after initializing the retriever
    # print(f"FAISS index dimension: {retriever.combined_faiss_index.d}")
    print(f"Embedding model dimension: {retriever.embedding_model.get_sentence_embedding_dimension()}")
    
    return retriever

# Usage example:
async def main():
    # Your processed documents from the preprocessing stage
    processor = AsyncDocumentProcessor()
    print(f"Preprocessing model name: {processor.embedding_model_name}")
    output_dir = Path("processed_documents")
    
    # Load indices from disk
    await processor.load_indices(str(output_dir))

    processed_documents = processor.documents # Your preprocessed documents
    # After loading your documents
    for doc_path, doc_data in processed_documents.items():
        for chunk in doc_data['chunks']:
            embedding = chunk['embedding']
            print(f"Document: {doc_path}")
            print(f"Chunk embedding shape: {np.array(embedding).shape}")
            break  # Just check the first chunk
        break  # Just check the first document

    api_key = os.getenv("ANTHROPIC_API_KEY")

    # Initialize the retriever
    retriever = await initialize_search_system(
        processed_documents=processed_documents,
        api_key=api_key
    )
    
    # Perform a search
    results, reasoning_steps = await retriever.search(
        "what are the use cases",
        return_reasoning=True
    )
    
    # Print results
    for result in results:
        print(f"Source: {result.source}")
        print(f"Text: {result.text}...")
        print(f"Score: {result.score}")
        print(f"Reasoning: {result.reasoning}\n")
    
    # for step in reasoning_steps:
    #     print(f"Step: {step}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())