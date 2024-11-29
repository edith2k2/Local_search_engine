import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from pathlib import Path
import json
import asyncio
import aiofiles
from dataclasses import dataclass, asdict
from tqdm.asyncio import tqdm_asyncio, tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from anthropic import AsyncAnthropic
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import PyPDF2
import logging
from typing import TypedDict

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchConfig(TypedDict):
    """Configuration for different batch processing parameters"""
    embeddings: int      # Number of texts to embed at once
    context: int         # Number of chunks for context generation
    faiss: int          # Number of vectors for FAISS updates
    documents: int       # Number of documents per main batch
    process: int        # Number of documents per process

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    file_path: str
    file_name: str
    file_type: str
    created_time: str
    modified_time: str
    size_bytes: int
    num_chunks: Optional[int] = None
    processing_time: Optional[float] = None
    batch_sizes: Optional[Dict[str, int]] = None

@dataclass
class ProcessedChunk:
    """Represents a processed document chunk with all associated data"""
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    metadata: dict
    context: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    tokenized_text: Optional[List[str]] = None

def initialize_worker():
    """Initialize necessary resources in each worker process"""
    global stop_words, model
    try:
        # Download NLTK resources silently
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
    except Exception as e:
        logger.error(f"Error initializing worker process: {str(e)}")
        raise

def compute_embeddings_worker(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    """Worker function for computing embeddings in separate processes"""
    try:
        # Initialize model in the worker process
        model = SentenceTransformer(model_name)
        # Handle device placement based on what's available
        if device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For Apple Silicon devices
            model = model.to('mps')
            # MPS requires explicit memory management
            torch.mps.empty_cache()
        elif device == 'cuda' and torch.cuda.is_available():
            # For NVIDIA GPUs
            model = model.to('cuda')
        else:
            # Fallback to CPU
            model = model.to('cpu')
            device = 'cpu'
        
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                device=device,
                batch_size=batch_size,
                show_progress_bar=False
            )
        return embeddings
        
    except Exception as e:
        logger.error(f"Error in embedding worker: {str(e)}")
        return np.zeros((len(texts), 768))  # Default embedding dimension
        
    finally:
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()

def preprocess_text_worker(text: str) -> List[str]:
    """Worker function for text preprocessing in separate processes"""
    try:
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in stop_words and token.isalpha()]
    except Exception as e:
        logger.error(f"Error in preprocessing worker: {str(e)}")
        return []

class AsyncDocumentProcessor:
    """
    Asynchronous document processor with multiprocessing support for efficient
    document processing, embedding computation, and index creation.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        anthropic_api_key: str = None,
        device: str = None,
        num_processes: int = None,
        batch_config: Optional[BatchConfig] = None
    ):
        """Initialize the document processor with multiprocessing support"""
        
        # Set up processing configuration
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info("Using NVIDIA GPU with CUDA")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                logger.info("Using Apple Silicon with Metal Performance Shaders")
            else:
                device = 'cpu'
                logger.info("Using CPU for computations")
        
        self.device = device
        if self.device == 'mps':
            # Configure default MPS allocator
            torch.mps.set_per_process_memory_fraction(0.7)
        logger.info(f"Selected device for computation: {self.device}")

        logger.info(f"Using device: {self.device}")
        
        # Initialize multiprocessing components
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Initializing with {self.num_processes} processes")
        
        # Create process pool with initialization
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.num_processes,
            initializer=initialize_worker
        )
        
        # Create thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="doc_processor"
        )
        
        # Set up batch configuration
        self.batch_config = batch_config or {
            'embeddings': 32,
            'context': 10,
            'faiss': 1000,
            'documents': 5,
            'process': max(1, self.num_processes // 2)
        }
        
        # Initialize model in main process for reference
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to(self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize other components
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Storage for processed documents
        self.documents: Dict[str, Dict] = {}

    async def read_file(self, file_path: str) -> tuple[List[Document], DocumentMetadata]:
        """Read file content and metadata with appropriate async handling"""
        path = Path(file_path)
        stats = path.stat()
        
        metadata = DocumentMetadata(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_type=path.suffix.lower(),
            created_time=datetime.fromtimestamp(stats.st_ctime).isoformat(),
            modified_time=datetime.fromtimestamp(stats.st_mtime).isoformat(),
            size_bytes=stats.st_size,
            batch_sizes=self.batch_config
        )
        
        if path.suffix.lower() == '.pdf':
            content = await self._read_pdf(str(path))
            documents = [Document(page_content=content, metadata={"source": str(path)})]
        else:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                documents = [Document(page_content=content, metadata={"source": str(path)})]
        
        return documents, metadata

    async def _read_pdf(self, file_path: str) -> str:
        """Read PDF content using thread pool to avoid blocking"""
        def read_in_thread():
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return ' '.join(page.extract_text() for page in pdf_reader.pages)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            read_in_thread
        )

    async def compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings using multiple processes for parallel processing"""
        if not texts:
            return np.array([])
        
        try:
            # Split texts into batches for parallel processing
            process_batch_size = len(texts) // self.num_processes + 1
            batches = [
                texts[i:i + process_batch_size]
                for i in range(0, len(texts), process_batch_size)
            ]
            
            # Create partial function with fixed arguments
            worker_func = partial(
                compute_embeddings_worker,
                model_name=self.embedding_model_name,
                device=self.device,
                batch_size=self.batch_config['embeddings']
            )
            
            # Compute embeddings in parallel
            embeddings_futures = [
                self.process_pool.submit(worker_func, batch)
                for batch in batches if batch
            ]
            
            # Gather results
            embeddings_list = [future.result() for future in embeddings_futures]
            
            # Combine results
            return np.vstack(embeddings_list)
            
        except Exception as e:
            logger.error(f"Error in parallel embedding computation: {str(e)}")
            return np.zeros((len(texts), self.embedding_dim))

    async def preprocess_texts_parallel(self, texts: List[str]) -> List[List[str]]:
        """Preprocess texts in parallel using multiple processes"""
        try:
            # Submit preprocessing tasks to process pool
            futures = [
                self.process_pool.submit(preprocess_text_worker, text)
                for text in texts
            ]
            
            # Gather results
            return [future.result() for future in futures]
            
        except Exception as e:
            logger.error(f"Error in parallel text preprocessing: {str(e)}")
            return [[] for _ in texts]

    async def generate_context_batch(
        self,
        chunks: List[ProcessedChunk],
        full_text: str
    ) -> List[str]:
        """Generate context descriptions for chunks using Claude"""
        batch_size = self.batch_config['context']
        all_contexts = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Create prompts for the entire batch
            prompts = []
            for chunk in batch_chunks:
                relative_pos = chunk.start_char / len(full_text)
                position_desc = "beginning" if relative_pos < 0.33 else \
                              "middle" if relative_pos < 0.66 else "end"
                
                context_window = 500
                prev_text = full_text[max(0, chunk.start_char - context_window):chunk.start_char]
                next_text = full_text[chunk.end_char:min(len(full_text), chunk.end_char + context_window)]
                
                prompt = f"""Generate a brief search-optimized context for this document chunk.
                Location: {position_desc} of document
                Previous text: {prev_text}
                Current chunk: {chunk.text}
                Following text: {next_text}
                Provide a concise context (50-75 tokens) that describes this chunk's key topics."""
                prompts.append(prompt)
            
            # Process batch with rate limiting
            async with asyncio.Semaphore(3):
                try:
                    response = await self.client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=75 * len(batch_chunks),
                        temperature=0.3,
                        messages=[{"role": "user", "content": "\n---\n".join(prompts)}]
                    )
                    
                    batch_contexts = response.content[0].text.strip().split("\n---\n")
                    all_contexts.extend(batch_contexts)
                    
                except Exception as e:
                    logger.error(f"Error in context generation batch {i}-{i+batch_size}: {str(e)}")
                    all_contexts.extend([f"[Context generation failed]"] * len(batch_chunks))
                
                await asyncio.sleep(0.1)
        
        return all_contexts

    def create_faiss_index_batch(
        self,
        embeddings: np.ndarray
    ) -> faiss.Index:
        """Create FAISS index in batches to manage memory"""
        batch_size = self.batch_config['faiss']
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size].astype('float32')
            index.add(batch)
        
        return index

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document with parallel processing support"""
        start_time = datetime.now()
        
        try:
            # Load document
            documents, metadata = await self.read_file(file_path)
            full_text = "\n\n".join(doc.page_content for doc in documents)
            
            # Split into chunks
            chunks = self.text_splitter.create_documents([full_text])
            
            # Create ProcessedChunk objects
            processed_chunks = []
            current_pos = 0
            for i, chunk in enumerate(chunks):
                chunk_start = full_text.find(chunk.page_content, current_pos)
                chunk_end = chunk_start + len(chunk.page_content)
                current_pos = chunk_end
                
                processed_chunks.append(ProcessedChunk(
                    chunk_id=i,
                    text=chunk.page_content,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata=chunk.metadata
                ))
            
            # Process chunks in parallel
            chunk_texts = [chunk.text for chunk in processed_chunks]
            tokenized_texts = await self.preprocess_texts_parallel(chunk_texts)
            
            for chunk, tokens in zip(processed_chunks, tokenized_texts):
                chunk.tokenized_text = tokens
            
            # Generate contexts
            # contexts = await self.generate_context_batch(processed_chunks, full_text)
            
            # # Update chunks with contexts
            # for chunk, context in zip(processed_chunks, contexts):
            #     chunk.context = context
            
            # Create enhanced texts and compute embeddings
            enhanced_texts = [f"{chunk.context}\n\n{chunk.text}" 
                            for chunk in processed_chunks]
            
            embeddings = await self.compute_embeddings_batch(enhanced_texts)
            
            # Create indices
            faiss_index = self.create_faiss_index_batch(embeddings)
            bm25_index = BM25Okapi([chunk.tokenized_text for chunk in processed_chunks])
            
            # Update metadata
            metadata.num_chunks = len(processed_chunks)
            metadata.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            result = {
                "metadata": asdict(metadata),
                "chunks": [asdict(chunk) for chunk in processed_chunks],
                "faiss_index": faiss_index,
                "bm25_index": bm25_index
            }

            self.documents[file_path] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    async def process_document_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of documents in parallel"""
        try:
            # Group documents into process batches
            process_batch_size = self.batch_config['process']
            batches = [
                file_paths[i:i + process_batch_size]
                for i in range(0, len(file_paths), process_batch_size)
            ]
            
            results = []
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")
                
                # Process documents in parallel
                batch_results = await asyncio.gather(
                    *(self.process_document(path) for path in batch),
                    return_exceptions=True
                )
                
                # Filter out exceptions and collect successful results
                for path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process {path}: {str(result)}")
                    else:
                        results.append(result)
                
                # Optional delay between batches
                await asyncio.sleep(0.1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing document batch: {str(e)}")
            return []

    async def process_documents(self, file_paths: List[str]) -> Dict[str, Dict]:
        """Process all documents with progress tracking and parallel processing"""
        logger.info(f"Starting processing of {len(file_paths)} documents "
                   f"using {self.num_processes} processes")
        
        try:
            # Process documents in batches
            results = []
            batch_size = self.batch_config['documents']

            with tqdm(total=len(file_paths), desc="Processing documents") as pbar:
                for i in range(0, len(file_paths), batch_size):
                    batch = file_paths[i:i + batch_size]
                    batch_results = await self.process_document_batch(batch)
                    results.extend(batch_results)
                    pbar.update(len(batch))
            
            # Convert results to dictionary
            return {
                result['metadata']['file_path']: result
                for result in results if result is not None
            }
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            return {}

    async def save_indices(self, output_dir: str):
        """Save processed documents and indices to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for file_path, doc_data in self.documents.items():
            doc_dir = output_path / Path(file_path).stem
            doc_dir.mkdir(exist_ok=True)
            
            try:
                # Save metadata
                async with aiofiles.open(doc_dir / 'metadata.json', 'w') as f:
                    await f.write(json.dumps(doc_data['metadata'], indent=2))
                
                # Handle numpy arrays for JSON serialization
                chunks_data = doc_data['chunks']
                for chunk in chunks_data:
                    if isinstance(chunk['embedding'], np.ndarray):
                        chunk['embedding'] = chunk['embedding'].tolist()
                
                # Save chunks
                async with aiofiles.open(doc_dir / 'chunks.json', 'w') as f:
                    await f.write(json.dumps(chunks_data, indent=2))
                
                # Save FAISS index
                faiss.write_index(doc_data['faiss_index'], 
                                str(doc_dir / 'faiss.index'))
                
                # Save BM25 index
                with open(doc_dir / 'bm25.pkl', 'wb') as f:
                    pickle.dump(doc_data['bm25_index'], f)
                
                logger.info(f"Successfully saved indices for {file_path}")
                
            except Exception as e:
                logger.error(f"Error saving indices for {file_path}: {str(e)}")

    async def load_indices(self, input_dir: str):
        """Load processed documents and indices from disk"""
        input_path = Path(input_dir)
        
        for doc_dir in input_path.iterdir():
            if doc_dir.is_dir():
                try:
                    # Load metadata
                    async with aiofiles.open(doc_dir / 'metadata.json', 'r') as f:
                        metadata = json.loads(await f.read())
                    
                    # Load chunks
                    async with aiofiles.open(doc_dir / 'chunks.json', 'r') as f:
                        chunks = json.loads(await f.read())
                    
                    # Convert lists back to numpy arrays
                    for chunk in chunks:
                        if isinstance(chunk['embedding'], list):
                            chunk['embedding'] = np.array(chunk['embedding'])
                    
                    # Load FAISS index
                    index = faiss.read_index(str(doc_dir / 'faiss.index'))
                    
                    # Load BM25 index
                    with open(doc_dir / 'bm25.pkl', 'rb') as f:
                        bm25_index = pickle.load(f)
                    
                    self.documents[metadata['file_path']] = {
                        "metadata": metadata,
                        "chunks": chunks,
                        "faiss_index": index,
                        "bm25_index": bm25_index
                    }
                    
                    logger.info(f"Successfully loaded indices for {metadata['file_path']}")
                    
                except Exception as e:
                    logger.error(f"Error loading indices from {doc_dir}: {str(e)}")

    def get_document_stats(self) -> Dict[str, Any]:
        """Generate comprehensive statistics about processed documents"""
        stats = {
            "total_documents": len(self.documents),
            "total_chunks": sum(doc["metadata"]["num_chunks"] for doc in self.documents.values()),
            "total_processing_time": sum(doc["metadata"]["processing_time"] for doc in self.documents.values()),
            "system_info": {
                "device": self.device,
                "num_processes": self.num_processes,
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dim,
                "batch_config": self.batch_config
            },
            "documents": {}
        }
        
        if stats["total_documents"] > 0:
            stats["average_chunks_per_doc"] = stats["total_chunks"] / stats["total_documents"]
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_documents"]
        
        for file_path, doc_data in self.documents.items():
            stats["documents"][file_path] = {
                "chunks": doc_data["metadata"]["num_chunks"],
                "processing_time": doc_data["metadata"]["processing_time"],
                "size_bytes": doc_data["metadata"]["size_bytes"],
                "file_type": doc_data["metadata"]["file_type"]
            }
        
        return stats

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        self.process_pool.shutdown(wait=True)
        self.thread_pool.shutdown(wait=True)
        if self.device == 'cuda':
            torch.cuda.empty_cache()