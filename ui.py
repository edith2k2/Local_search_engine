import streamlit as st
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
from anthropic import AsyncAnthropic
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our search components
from preprocessing import AsyncDocumentProcessor, BatchConfig
from retriever import ChainOfThoughtRetriever, SearchResult
from answer_generator import AnswerGenerator, GeneratedAnswer

# Enable nested async support for Streamlit
nest_asyncio.apply()

class StreamlitSearchUI:
    def __init__(self):
        # Configure the page
        st.set_page_config(
            page_title="Document Search",
            page_icon="🔍",
            layout="wide"
        )
        
        # Initialize session state
        self._initialize_session_state()
        
        # Set up async event loop and thread executor
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load configuration
        self.load_configuration()
        
        # Initialize components if API key exists
        if st.session_state.api_key and not st.session_state.components_ready:
            self._run_async(self._ensure_components_initialized())

    def _initialize_session_state(self):
        """Initialize session state variables"""
        default_state = {
            'search_results': [],
            'documents': {},
            'components_ready': False,
            'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
            'processor': None,
            'retriever': None,
            'session_initialized': True,
            'processing_status': None,
            'error_message': None
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def load_configuration(self):
        """Load system configuration"""
        # Determine optimal device
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        
        self.config = {
            'embedding_model': "sentence-transformers/all-mpnet-base-v2",
            'index_directory': Path("processed_documents"),
            'device': device,
            'batch_config': BatchConfig(
                embeddings=32,
                context=10,
                faiss=1000,
                documents=5,
                process=4
            )
        }
        
        # Ensure index directory exists
        self.config['index_directory'].mkdir(parents=True, exist_ok=True)

    def _run_async(self, coro):
        """Run async code in the thread executor"""
        return self.loop.run_until_complete(coro)

    async def _ensure_components_initialized(self):
        """Initialize search components with error handling"""
        try:
            if not st.session_state.api_key:
                return
            
            # Create processor instance
            processor = AsyncDocumentProcessor(
                embedding_model_name=self.config['embedding_model'],
                anthropic_api_key=st.session_state.api_key,
                device=self.config['device'],
                batch_config=self.config['batch_config']
            )
            
            # Load existing indices
            if self.config['index_directory'].exists():
                await processor.load_indices(str(self.config['index_directory']))
                st.session_state.documents = processor.documents
            
            # Initialize retriever
            retriever = ChainOfThoughtRetriever(
                documents=st.session_state.documents,
                embedding_model=processor.embedding_model,
                anthropic_client=processor.client,
                device=self.config['device']
            )
            
            # Update session state
            st.session_state.processor = processor
            st.session_state.retriever = retriever
            st.session_state.components_ready = True
            st.session_state.error_message = None
            
        except Exception as e:
            st.session_state.error_message = f"Initialization error: {str(e)}"
            st.session_state.components_ready = False

    async def _process_documents(self, uploaded_files):
        """Process uploaded documents"""
        if not uploaded_files:
            return
        
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        file_paths = []
        
        try:
            # Save uploads temporarily
            for uploaded_file in uploaded_files:
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(str(temp_path))
            
            # Process documents
            st.session_state.processing_status = "Processing documents..."
            processed_docs = await st.session_state.processor.process_documents(file_paths)
            
            # Update storage
            st.session_state.documents.update(processed_docs)
            await st.session_state.processor.save_indices(str(self.config['index_directory']))
            
            st.session_state.processing_status = f"Successfully processed {len(processed_docs)} documents!"
            
        except Exception as e:
            st.session_state.error_message = f"Processing error: {str(e)}"
            
        finally:
            # Cleanup
            for path in file_paths:
                Path(path).unlink(missing_ok=True)
            if temp_dir.exists():
                temp_dir.rmdir()

    async def _perform_search(self, query: str) -> List[SearchResult]:
        """Execute search query"""
        try:
            results, steps = await st.session_state.retriever.search(query)
            return results
        except Exception as e:
            st.session_state.error_message = f"Search error: {str(e)}"
            return []

    def render_ui(self):
        """Render the main user interface"""
        st.title("Document Search and Processing System")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # API key input
            if not st.session_state.api_key:
                api_key = st.text_input("Enter Anthropic API Key", type="password")
                if api_key:
                    st.session_state.api_key = api_key
                    self._run_async(self._ensure_components_initialized())
            
            # Document upload section
            if st.session_state.components_ready:
                st.header("Document Processing")
                uploaded_files = st.file_uploader(
                    "Upload new documents",
                    accept_multiple_files=True,
                    type=['txt', 'pdf']
                )
                
                if uploaded_files and st.button("Process Documents"):
                    self._run_async(self._process_documents(uploaded_files))
                
                if st.session_state.processing_status:
                    st.info(st.session_state.processing_status)
        
        # Main search interface
        if st.session_state.components_ready:
            st.markdown("### Search")
            query = st.text_input(
                "Enter your search query",
                key="search_input",
                placeholder="What would you like to know?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                search_clicked = st.button("Search", type="primary")
            
            if search_clicked and query:
                with st.spinner("Searching documents..."):
                    results = self._run_async(self._perform_search(query))
                    if results:
                        st.session_state.search_results = results
            
            # Display results
            if st.session_state.search_results:
                st.markdown("### Results")
                for i, result in enumerate(st.session_state.search_results, 1):
                    with st.expander(f"Result {i} - {Path(result.source).name}"):
                        st.markdown(result.text)
                        if result.context:
                            st.markdown(f"*Context: {result.context}*")
                        st.markdown(f"*Relevance Score: {result.score:.2f}*")
        
        # Error display
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            if st.button("Clear Error"):
                st.session_state.error_message = None

    def run(self):
        """Main application entry point"""
        try:
            self.render_ui()
        finally:
            # Cleanup
            self.executor.shutdown(wait=False)

if __name__ == "__main__":
    search_ui = StreamlitSearchUI()
    search_ui.run()