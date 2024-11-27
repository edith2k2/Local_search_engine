import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from pathlib import Path
import json
import asyncio
import aiofiles
from dataclasses import dataclass, asdict
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from anthropic import AsyncAnthropic
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

@dataclass
class DocumentMetadata:
    """Stores metadata about processed documents"""
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
    """Represents a processed document chunk with its metadata"""
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    metadata: dict
    context: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    tokenized_text: Optional[List[str]] = None  # For BM25

class DocumentProcessor:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 500,     # Characters per chunk
        chunk_overlap: int = 50,   # Character overlap between chunks
        anthropic_api_key: str = None,
        device: str = None
    ):
        # Download required NLTK resources
        for resource in ['punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)
        
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize embedding model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for embeddings")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to(self.device)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Anthropic client for context generation
        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        
        # Initialize dimensions
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Storage for processed documents
        self.documents: Dict[str, Dict] = {}

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words and token.isalpha()]
        return tokens

    async def read_file(self, file_path: str) -> tuple[List[Document], DocumentMetadata]:
        """Read file content and metadata using LangChain loaders"""
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
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path))
            
        documents = loader.load()
        return documents, metadata

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts"""
        try:
            with torch.no_grad():
                embeddings = self.embedding_model.encode(
                    texts,
                    normalize_embeddings=True,
                    device=self.device,
                    batch_size=32
                )
            return embeddings
        except Exception as e:
            print(f"Error computing embeddings: {str(e)}")
            return np.zeros((len(texts), self.embedding_dim))
        finally:
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    async def generate_context_batch(self, chunks: List[ProcessedChunk], full_text: str) -> List[str]:
        """Generate context descriptions for chunks using Claude"""
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
        async def process_chunk(chunk: ProcessedChunk) -> str:
            try:
                relative_pos = chunk.start_char / len(full_text)
                position_desc = "beginning" if relative_pos < 0.33 else \
                              "middle" if relative_pos < 0.66 else "end"
                
                context_window = 500
                prev_text = full_text[max(0, chunk.start_char - context_window):chunk.start_char]
                next_text = full_text[chunk.end_char:min(len(full_text), chunk.end_char + context_window)]

                prompt = f"""Generate a brief search-optimized context for this document chunk.

Location: {position_desc} of document

Previous text:
{prev_text}

Current chunk:
{chunk.text}

Following text:
{next_text}

Provide a concise context (50-75 tokens) that describes this chunk's key topics and role in the document."""

                response = await self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=75,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text.strip()
            
            except Exception as e:
                print(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
                return f"[Context generation failed for chunk {chunk.chunk_id}]"

        sem = asyncio.Semaphore(3)
        
        async def process_with_semaphore(chunk: ProcessedChunk) -> str:
            async with sem:
                return await process_chunk(chunk)
        
        contexts = await asyncio.gather(
            *(process_with_semaphore(chunk) for chunk in chunks)
        )
        
        return contexts

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document including BM25 and FAISS indexing"""
        start_time = datetime.now()
        
        try:
            # Load document
            documents, metadata = await self.read_file(file_path)
            full_text = "\n\n".join(doc.page_content for doc in documents)
            
            # Split into chunks using LangChain
            chunks = self.text_splitter.create_documents(
                [full_text],
                metadatas=[{"source": file_path}]
            )
            
            # Convert to ProcessedChunk objects
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
                    metadata=chunk.metadata,
                    tokenized_text=self.preprocess_text(chunk.page_content)
                ))
            
            metadata.num_chunks = len(processed_chunks)
            
            # Generate contexts
            contexts = await self.generate_context_batch(processed_chunks, full_text)
            for chunk, context in zip(processed_chunks, contexts):
                chunk.context = context
            
            # Create enhanced texts with context
            enhanced_texts = [f"{chunk.context}\n\n{chunk.text}" for chunk in processed_chunks]
            
            # Compute embeddings for FAISS
            embeddings = self.compute_embeddings(enhanced_texts)
            
            # Create FAISS index
            index = faiss.IndexFlatL2(self.embedding_dim)
            index.add(embeddings.astype('float32'))
            
            # Create BM25 index
            tokenized_texts = [self.preprocess_text(text) for text in enhanced_texts]
            bm25_index = BM25Okapi(tokenized_texts)
            
            # Store embeddings with chunks
            for chunk, embedding in zip(processed_chunks, embeddings):
                chunk.embedding = embedding
            
            metadata.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            result = {
                "metadata": asdict(metadata),
                "chunks": [asdict(chunk) for chunk in processed_chunks],
                "faiss_index": index,
                "bm25_index": bm25_index
            }
            
            self.documents[file_path] = result
            return result
            
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
            raise

    async def process_documents(self, file_paths: List[str]) -> Dict[str, Dict]:
        """Process multiple documents in parallel"""
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
        """Save processed documents and indices"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for file_path, doc_data in self.documents.items():
            doc_dir = output_path / Path(file_path).stem
            doc_dir.mkdir(exist_ok=True)
            
            # Save metadata and chunks
            async with aiofiles.open(doc_dir / 'metadata.json', 'w') as f:
                await f.write(json.dumps(doc_data['metadata'], indent=2))
            
            # Convert numpy arrays to lists for JSON serialization
            chunks_data = doc_data['chunks']
            for chunk in chunks_data:
                if isinstance(chunk['embedding'], np.ndarray):
                    chunk['embedding'] = chunk['embedding'].tolist()
            
            async with aiofiles.open(doc_dir / 'chunks.json', 'w') as f:
                await f.write(json.dumps(chunks_data, indent=2))
            
            # Save FAISS index
            faiss.write_index(doc_data['faiss_index'], 
                            str(doc_dir / 'faiss.index'))
            
            # Save BM25 index
            with open(doc_dir / 'bm25.pkl', 'wb') as f:
                pickle.dump(doc_data['bm25_index'], f)

    async def load_indices(self, input_dir: str):
        """Load processed documents and indices"""
        input_path = Path(input_dir)
        
        for doc_dir in input_path.iterdir():
            if doc_dir.is_dir():
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