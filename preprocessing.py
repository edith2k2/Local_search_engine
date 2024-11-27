import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
# from transformers import AutoTokenizer, AutoModel
import torch
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from anthropic import AsyncAnthropic
import pandas as pd
from pathlib import Path
import PyPDF2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
import json
from dataclasses import dataclass, asdict
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from functools import partial



resources = ['punkt', 'punkt_tab']

# Download the resources if not already present
for resource in resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
        print(f"{resource} is already downloaded.")
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource)

@dataclass
class DocumentMetadata:
    file_path: str
    file_name: str
    file_type: str
    created_time: str
    modified_time: str
    size_bytes: int
    num_chunks: Optional[int] = None
    processing_time: Optional[float] = None

@dataclass
class ProcessedChunk:
    chunk_id: int
    text: str
    start_idx: int
    end_idx: int
    context: Optional[str] = None
    embedding: Optional[np.ndarray] = None


def compute_embeddings_batch(texts: List[str], model_name: str, device: str) -> List[np.ndarray]:
    """Compute embeddings for a batch of texts using SentenceTransformer with specified device."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer(model_name)
    model = model.to(device)  # Move model to specified device
    try:
        with torch.no_grad():  # Disable gradient computation for inference
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                device=device,
                batch_size=32  # Adjust based on GPU memory
            )
        return embeddings
    except Exception as e:
        print(f"Error computing embeddings batch: {str(e)}")
        return [np.zeros(model.get_sentence_embedding_dimension()) for _ in texts]
    finally:
        # Clean up GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()

class DocumentProcessor:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        anthropic_api_key: str = None,
        batch_size: int = 5,
        max_workers: int = None,
        device: str = None
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.embedding_model_name = embedding_model_name
        
        # Set device for embeddings
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for embeddings")
        
        # Initialize SentenceTransformer
        self.model = SentenceTransformer(embedding_model_name)
        self.model = self.model.to(self.device)
        
        # Use model's tokenizer for chunk size estimation
        self.tokenizer = self.model.tokenizer
        
        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        
        # Initialize thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=self._init_worker  # Initialize worker processes
        )
        
        # Storage for processed documents
        self.documents: Dict[str, Dict] = {}
    
    @staticmethod
    def _init_worker():
        """Initialize worker process with proper environment settings."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def read_file(self, file_path: str) -> tuple[str, DocumentMetadata]:
        """Read file content and metadata asynchronously."""
        path = Path(file_path)
        stats = path.stat()
        
        metadata = DocumentMetadata(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_type=path.suffix.lower(),
            created_time=datetime.fromtimestamp(stats.st_ctime).isoformat(),
            modified_time=datetime.fromtimestamp(stats.st_mtime).isoformat(),
            size_bytes=stats.st_size
        )
        
        if path.suffix.lower() == '.pdf':
            # PDF reading needs to be done in a thread pool as PyPDF2 is not async
            content = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self._read_pdf,
                file_path
            )
        else:
            # For text files, use async reading
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        
        return content, metadata

    def _read_pdf(self, file_path: str) -> str:
        """Read PDF file content (runs in thread pool)."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() for page in pdf_reader.pages)

    def visualize_tokens(self, text: str) -> tuple[str, int]:
        """Helper function to show how text is tokenized and count tokens."""
        tokens = self.tokenizer.tokenize(text)
        return f"Text: '{text}'\nTokens ({len(tokens)}): {tokens}"

    def chunk_document(self, document: str) -> List[ProcessedChunk]:
        """
        Split document into chunks based on token count (not character length).
        chunk_size: maximum number of tokens per chunk
        chunk_overlap: number of tokens to overlap between chunks
        """
        sentences = sent_tokenize(document)
        chunks = []
        current_chunk = []
        current_token_count = 0  # Track number of tokens
        chunk_start_idx = 0
        chunk_id = 0
        
        # Debug information
        print(f"Target chunk size: {self.chunk_size} tokens")
        print(f"Target overlap: {self.chunk_overlap} tokens")
        
        for sentence in sentences:
            # Count tokens in this sentence
            sentence_tokens = self.tokenizer(sentence, return_tensors="pt")
            sentence_token_count = len(sentence_tokens['input_ids'][0])
            
            print(f"\nProcessing sentence: '{sentence}'")
            print(f"Token count: {sentence_token_count}")
            
            # Handle sentences that are longer than chunk_size
            if sentence_token_count > self.chunk_size:
                print(f"⚠️ Long sentence detected: {sentence_token_count} tokens > {self.chunk_size} chunk_size")
                
                # Store any accumulated sentences first
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    token_count = len(self.tokenizer(chunk_text, return_tensors="pt")['input_ids'][0])
                    chunks.append(ProcessedChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start_idx=chunk_start_idx,
                        end_idx=chunk_start_idx + len(chunk_text)
                    ))
                    print(f"Created chunk {chunk_id} with {token_count} tokens")
                    chunk_id += 1
                
                # Handle the long sentence as its own chunk
                long_sent_start = document.find(sentence)
                chunks.append(ProcessedChunk(
                    chunk_id=chunk_id,
                    text=sentence,
                    start_idx=long_sent_start,
                    end_idx=long_sent_start + len(sentence)
                ))
                print(f"Created chunk {chunk_id} for long sentence with {sentence_token_count} tokens")
                
                chunk_id += 1
                current_chunk = []
                current_token_count = 0
                chunk_start_idx = long_sent_start + len(sentence)
                continue
                
            # Check if adding this sentence would exceed chunk_size
            if current_token_count + sentence_token_count > self.chunk_size and current_chunk:
                # Store current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(ProcessedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_idx=chunk_start_idx,
                    end_idx=chunk_start_idx + len(chunk_text)
                ))
                print(f"Created chunk {chunk_id} with {current_token_count} tokens")
                
                # Calculate overlap for next chunk
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                overlap_sentences = current_chunk[overlap_start:]
                overlap_text = " ".join(overlap_sentences)
                overlap_tokens = len(self.tokenizer(overlap_text, return_tensors="pt")['input_ids'][0])
                
                print(f"Overlap text: '{overlap_text}'")
                print(f"Overlap tokens: {overlap_tokens}")
                
                # Reset for next chunk
                chunk_id += 1
                current_chunk = overlap_sentences + [sentence]
                chunk_start_idx = chunk_start_idx + len(chunk_text) - len(overlap_text)
                current_token_count = len(self.tokenizer(" ".join(current_chunk), return_tensors="pt")['input_ids'][0])
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_token_count = len(self.tokenizer(" ".join(current_chunk), return_tensors="pt")['input_ids'][0])
                print(f"Added sentence. Current token count: {current_token_count}")
        
        # Add final chunk if there are remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            final_tokens = len(self.tokenizer(chunk_text, return_tensors="pt")['input_ids'][0])
            if final_tokens > 0:
                chunks.append(ProcessedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_idx=chunk_start_idx,
                    end_idx=chunk_start_idx + len(chunk_text)
                ))
                print(f"Created final chunk {chunk_id} with {final_tokens} tokens")
        
        return chunks

    async def generate_context_batch(self, chunks: List[ProcessedChunk], document: str) -> List[str]:
        """Generate context for a batch of chunks using Claude Haiku."""
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=6)
        )
        async def process_chunk(chunk: ProcessedChunk) -> str:
            try:
                # Calculate relative position
                doc_length = len(document)
                relative_pos = chunk.start_idx / doc_length
                position_desc = "beginning" if relative_pos < 0.33 else \
                            "middle" if relative_pos < 0.66 else "end"
                
                # Get surrounding context (reduced window for efficiency)
                context_window = 500
                prev_text = document[max(0, chunk.start_idx - context_window):chunk.start_idx]
                next_text = document[chunk.end_idx:min(doc_length, chunk.end_idx + context_window)]

                # Simplified prompt for Haiku
                prompt = f"""You are helping to generate search-optimized context for document chunks. 
Given a portion of a document with its surrounding context, create a brief contextual description.

Location: {position_desc} of document

Previous text:
<context>{prev_text}</context>

Current chunk:
<chunk>{chunk.text}</chunk>

Following text:
<context>{next_text}</context>

Provide a concise context (50-75 tokens) that situates this chunk within the document and highlights key topics for search retrieval. Answer with only the context description."""

                response = await self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=75,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text.strip()
            
            except Exception as e:
                print(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
                raise

        async def process_batch(batch: List[ProcessedChunk]) -> List[str]:
            tasks = []
            for chunk in batch:
                tasks.append(process_chunk(chunk))
                await asyncio.sleep(0.05)  # 50ms delay between requests
            return await asyncio.gather(*tasks)

        # Process chunks in batches
        batch_size = min(10, len(chunks))
        contexts = []
        
        # Create semaphore to limit concurrent batches
        sem = asyncio.Semaphore(3)  # Limit to 3 concurrent batches
        
        async def process_batch_with_semaphore(batch: List[ProcessedChunk]) -> List[str]:
            async with sem:
                return await process_batch(batch)
        
        # Create all batch processing tasks
        tasks = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            tasks.append(process_batch_with_semaphore(batch))
        
        # Process all batches with gathered results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        for i, result in enumerate(results):
            batch_start = i * batch_size
            batch_size_current = min(batch_size, len(chunks) - batch_start)
            
            if isinstance(result, Exception):
                print(f"Error processing batch starting at chunk {batch_start}: {str(result)}")
                contexts.extend([f"[Context generation failed for chunk {j}]" 
                            for j in range(batch_start, batch_start + batch_size_current)])
            else:
                contexts.extend(result)
        
        return contexts

    async def process_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Process embeddings in batches using process pool or GPU directly."""
        if self.device == 'cpu':
            # Use process pool for CPU computations
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    self.process_pool,
                    partial(
                        compute_embeddings_batch,
                        model_name=self.embedding_model_name,
                        device=self.device
                    ),
                    batch_texts
                )
                embeddings.extend(batch_embeddings)
            
            return embeddings
        else:
            # Use GPU directly without process pool
            try:
                with torch.no_grad():
                    embeddings = self.model.encode(
                        texts,
                        normalize_embeddings=True,
                        device=self.device,
                        batch_size=32  # Adjust based on your GPU memory
                    )
                return embeddings
            except Exception as e:
                print(f"Error computing embeddings: {str(e)}")
                return [np.zeros(self.model.get_sentence_embedding_dimension()) 
                        for _ in texts]
            finally:
                # Clean up GPU memory
                torch.cuda.empty_cache()

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document including reading, chunking, context generation,
        and index creation."""
        start_time = datetime.now()
        
        try:
            # Read document and metadata
            content, metadata = await self.read_file(file_path)
            
            # Chunk document
            chunks = self.chunk_document(content)
            metadata.num_chunks = len(chunks)
            
            # Generate contexts
            # contexts = await self.generate_context_batch(chunks, content)
            # for chunk, context in zip(chunks, contexts):
            #     chunk.context = context
            
            # Compute embeddings for enhanced texts
            enhanced_texts = [f"{chunk.context}\n\n{chunk.text}" for chunk in chunks]
            embeddings = await self.process_embeddings(enhanced_texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Create BM25 index
            tokenized_chunks = [word_tokenize(text.lower()) for text in enhanced_texts]
            bm25 = BM25Okapi(tokenized_chunks)
            
            # Calculate processing time
            metadata.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            result = {
                "metadata": asdict(metadata),
                "chunks": [asdict(chunk) for chunk in chunks],
                "bm25_index": bm25
            }
            
            self.documents[file_path] = result
            return result
            
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
            raise

    async def process_documents(self, file_paths: List[str]) -> Dict[str, Dict]:
        """Process multiple documents in parallel."""
        async def process_with_progress(file_path: str):
            try:
                return await self.process_document(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return None

        results = await tqdm_asyncio.gather(
            *(process_with_progress(file_path) for file_path in file_paths),
            desc="Processing documents"
        )
        
        return {
            file_path: result 
            for file_path, result in zip(file_paths, results) 
            if result is not None
        }

    async def save_indices(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for file_path, doc_data in self.documents.items():
            doc_dir = output_path / Path(file_path).stem
            doc_dir.mkdir(exist_ok=True)
            
            # Save metadata and chunks
            async with aiofiles.open(doc_dir / 'metadata.json', 'w') as f:
                await f.write(json.dumps(doc_data['metadata'], indent=2))
            
            chunks_data = doc_data['chunks']
            for chunk in chunks_data:
                if isinstance(chunk['embedding'], np.ndarray):
                    chunk['embedding'] = chunk['embedding'].tolist()
            
            async with aiofiles.open(doc_dir / 'chunks.json', 'w') as f:
                await f.write(json.dumps(chunks_data, indent=2))
            
            # Save embeddings
            embeddings = np.array([chunk['embedding'] for chunk in doc_data['chunks']])
            np.save(doc_dir / 'embeddings.npy', embeddings)

    async def load_indices(self, input_dir: str):
        input_path = Path(input_dir)
        
        for doc_dir in input_path.iterdir():
            if doc_dir.is_dir():
                async with aiofiles.open(doc_dir / 'metadata.json', 'r') as f:
                    metadata = json.loads(await f.read())
                
                async with aiofiles.open(doc_dir / 'chunks.json', 'r') as f:
                    chunks = json.loads(await f.read())
                
                embeddings = np.load(doc_dir / 'embeddings.npy')
                for chunk, embedding in zip(chunks, embeddings):
                    chunk['embedding'] = embedding
                
                # Reconstruct BM25 from chunks
                enhanced_texts = [f"{chunk['context']}\n\n{chunk['text']}" for chunk in chunks]
                tokenized_chunks = [word_tokenize(text.lower()) for text in enhanced_texts]
                bm25 = BM25Okapi(tokenized_chunks)
                
                self.documents[metadata['file_path']] = {
                    "metadata": metadata,
                    "chunks": chunks,
                    "bm25_index": bm25
                }