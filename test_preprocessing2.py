import asyncio
import logging
from pathlib import Path
# from temp3 import AsyncDocumentProcessor, logger
from preprocessing import AsyncDocumentProcessor, logger

async def main():
    # Configure batch sizes based on system capabilities
    batch_config = {
        'embeddings': 32,     # Number of texts to embed at once
        'context': 10,        # Number of chunks for context generation
        'faiss': 1000,       # Number of vectors for FAISS updates
        'documents': 5,       # Number of documents per main batch
        'process': 2         # Number of documents per process batch
    }
    
    # Initialize the processor with multiprocessing support
    async with AsyncDocumentProcessor(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        chunk_size=500,
        chunk_overlap=50,
        anthropic_api_key="your-api-key-here",
        num_processes=4,  # Adjust based on available CPU cores
        batch_config=batch_config,
        device='mps'
    ) as processor:
        try:
            # Define document paths
            documents_path = Path("pdfs")
            file_paths = [
                str(f) for f in documents_path.iterdir() 
                if f.suffix.lower() in {'.pdf', '.txt'}
            ]
            
            # Process all documents
            logger.info("Starting document processing...")
            results = await processor.process_documents(file_paths)
            
            # Save processed documents and indices
            output_dir = Path("processed_documents")
            logger.info(f"Saving indices to {output_dir}...")
            await processor.save_indices(str(output_dir))
            
            # Get processing statistics
            stats = processor.get_document_stats()
            
            # Log detailed statistics
            logger.info("\nProcessing Statistics:")
            logger.info(f"Total documents processed: {stats['total_documents']}")
            logger.info(f"Total chunks created: {stats['total_chunks']}")
            if stats['total_documents'] > 0:
                logger.info(f"Average chunks per document: {stats['average_chunks_per_doc']:.2f}")
                logger.info(f"Average processing time per document: {stats['average_processing_time']:.2f} seconds")
            logger.info(f"Total processing time: {stats['total_processing_time']:.2f} seconds")
            
            # Log system information
            logger.info("\nSystem Information:")
            logger.info(f"Device: {stats['system_info']['device']}")
            logger.info(f"Number of processes: {stats['system_info']['num_processes']}")
            logger.info(f"Embedding model: {stats['system_info']['embedding_model']}")
            logger.info(f"Batch configuration: {stats['system_info']['batch_config']}")
            
            # Optionally, load indices later
            logger.info("\nLoading saved indices...")
            await processor.load_indices(str(output_dir))
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the processor
    asyncio.run(main())