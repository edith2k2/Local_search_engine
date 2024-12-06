# Local Search Engine

A sophisticated local search engine implementation that combines dense and sparse retrieval with chain-of-thought reasoning for enhanced document search capabilities.

## Features

- **Hybrid Search**: Combines dense (embedding-based) and sparse (BM25) retrieval methods
- **Chain-of-Thought Reasoning**: Uses LLM-powered iterative refinement for improved search accuracy
- **Temporal Search**: Supports both UI-based and natural language temporal queries
- **Advanced Query Processing**: Includes query classification, spell correction, and intent analysis
- **Rich UI**: Built with Streamlit for an intuitive user experience
- **Document Processing**: Handles both text and PDF documents with automatic chunking
- **Asynchronous Processing**: Efficient document processing with multiprocessing support

## Architecture

### 1. Preprocessing Pipeline
- Context Generation: Uses LLM to generate chunk-specific context
- Index Creation:
  - BM25 index for sparse retrieval
  - Embeddings for dense retrieval
- Goal: Improves search relevance by enhancing chunk interpretability

### 2. Retrieval System
- Chain-of-Thought (CoT) Retrieval:
  - Step-by-step reasoning
  - Iterative refinement based on previous results
- Hybrid Retrieval:
  - Combines sparse and dense methods
  - Weighted fusion based on query type

### 3. Post-Processing
- Rank Fusion: Merges results from multiple retrieval methods
- Temporal Scoring: Adjusts relevance based on document timestamps
- Result Deduplication: Removes redundant content

### 4. Answer Generation
- Generates concise responses using LLM
- Includes supporting evidence and citations
- Confidence scoring for generated answers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-search-engine.git
cd local-search-engine
```

2. Create and activate a Python 3.9 virtual environment:
```bash
# For Unix/macOS
python3.9 -m venv venv
source venv/bin/activate

# For Windows
python3.9 -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r all_requirements.txt
```

## Configuration

The system requires the following environment variables:
```bash
ANTHROPIC_API_KEY=your_api_key
```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run ui.py
```

2. Upload documents through the UI or process them programmatically:
```python
from preprocessing import AsyncDocumentProcessor

processor = AsyncDocumentProcessor()
await processor.process_documents(['path/to/document.pdf'])
```

3. Perform searches:
```python
from retriever import ChainOfThoughtRetriever

retriever = ChainOfThoughtRetriever(documents=processor.documents)
results = await retriever.search("your query here")
```

## Components

### Query Parser (`query_parser.py`)
- Handles temporal expressions in queries
- Supports various time frames and constraints
- Provides flexible and strict temporal matching

### Query Classifier (`query_classifier.py`)
- Classifies query intent
- Performs spell correction
- Adjusts retrieval weights based on query type

### Preprocessor (`preprocessing.py`)
- Handles document ingestion and chunking
- Generates embeddings and indices
- Supports parallel processing

### Retriever (`retriever.py`)
- Implements the core search logic
- Manages chain-of-thought reasoning
- Handles result fusion and ranking

### Answer Generator (`answer_generator.py`)
- Generates natural language answers
- Provides citations and confidence scores
- Handles evidence integration

### UI (`ui.py`)
- Streamlit-based user interface
- Real-time search and processing
- Result visualization and exploration

## Dependencies

- `anthropic`: For LLM integration
- `sentence-transformers`: For embedding generation
- `faiss-cpu`: For efficient similarity search
- `rank_bm25`: For sparse retrieval
- `streamlit`: For the web interface
- `PyPDF2`: For PDF processing
- `spacy`: For text processing
- `torch`: For neural computations
- `numpy`: For efficient computation
- `langchain`: For efficient chunking
