import streamlit as st
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from sentence_transformers import SentenceTransformer
from anthropic import AsyncAnthropic
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Import our search components
from preprocessing import AsyncDocumentProcessor, BatchConfig
from retriever import ChainOfThoughtRetriever, SearchResult
from answer_generator import AnswerGenerator, GeneratedAnswer

# Enable nested async support for Streamlit
nest_asyncio.apply()

class StreamlitSearchUI:
    """
    A comprehensive Streamlit-based user interface for document search and answer generation.
    Handles document processing, search, and answer generation with a clean, intuitive interface.
    """
    
    def __init__(self):
        # Configure the page with a clean, professional layout
        st.set_page_config(
            page_title="AI-Powered Document Search",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state for persistent data
        self._initialize_session_state()
        
        # Set up async event loop and thread executor
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load system configuration
        self.load_configuration()
        
        # Initialize components if API key exists
        if st.session_state.api_key and not st.session_state.components_ready:
            self._run_async(self._initialize_components())

    def _initialize_session_state(self):
        """Initialize all required session state variables with default values"""
        default_state = {
            'search_results': None,  # Will store (results, generated_answer) tuple
            'documents': {},
            'components_ready': False,
            'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
            'processor': None,
            'retriever': None,
            'answer_generator': None,
            'session_initialized': True,
            'processing_status': None,
            'error_message': None,
            'last_query': None,
            'processing_time': None
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def load_configuration(self):
        """Load system configuration with optimal settings for the current environment"""
        # Determine the best available device
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
        """Run asynchronous code in the thread executor"""
        return self.loop.run_until_complete(coro)

    async def _initialize_components(self):
        """Initialize all required components with proper error handling"""
        try:
            if not st.session_state.api_key:
                return
            
            # Create document processor
            processor = AsyncDocumentProcessor(
                embedding_model_name=self.config['embedding_model'],
                anthropic_api_key=st.session_state.api_key,
                device=self.config['device'],
                batch_config=self.config['batch_config']
            )
            
            # Load existing document indices
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
            
            # Initialize answer generator
            answer_generator = AnswerGenerator(
                anthropic_client=processor.client
            )
            
            # Update session state
            st.session_state.processor = processor
            st.session_state.retriever = retriever
            st.session_state.answer_generator = answer_generator
            st.session_state.components_ready = True
            st.session_state.error_message = None
            
        except Exception as e:
            st.session_state.error_message = f"Initialization error: {str(e)}"
            st.session_state.components_ready = False

    async def _process_documents(self, uploaded_files):
        """Process uploaded documents with progress tracking"""
        if not uploaded_files:
            return
        
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        file_paths = []
        
        try:
            # Save uploaded files temporarily
            for uploaded_file in uploaded_files:
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(str(temp_path))
            
            # Process documents with status updates
            st.session_state.processing_status = "Processing documents..."
            processed_docs = await st.session_state.processor.process_documents(file_paths)
            
            # Update storage
            st.session_state.documents.update(processed_docs)
            await st.session_state.processor.save_indices(str(self.config['index_directory']))
            
            st.session_state.processing_status = (
                f"Successfully processed {len(processed_docs)} documents! "
                f"Total documents: {len(st.session_state.documents)}"
            )
            
        except Exception as e:
            st.session_state.error_message = f"Processing error: {str(e)}"
            
        finally:
            # Cleanup temporary files
            for path in file_paths:
                Path(path).unlink(missing_ok=True)
            if temp_dir.exists():
                temp_dir.rmdir()

    async def _perform_search(
        self,
        query: str
    ) -> Optional[Tuple[List[SearchResult], GeneratedAnswer]]:
        """Execute search and generate answer"""
        try:
            # Get search results
            start_time = time.time()
            results, retrieval_steps = await st.session_state.retriever.search(query)
            
            if not results:
                return None
            
            # Generate answer from results
            generated_answer = await st.session_state.answer_generator.generate_answer(
                query,
                results
            )
            
            # Record processing time
            st.session_state.processing_time = time.time() - start_time
            
            return results, generated_answer
            
        except Exception as e:
            st.session_state.error_message = f"Search error: {str(e)}"
            return None

    def _render_answer_section(self, generated_answer: GeneratedAnswer):
        """Render the generated answer section with citations"""
        st.markdown("### AI-Generated Answer")
        
        # Display confidence indicator
        confidence_color = (
            "🟢" if generated_answer.confidence_score >= 0.8 else
            "🟡" if generated_answer.confidence_score >= 0.5 else
            "🔴"
        )
        st.markdown(
            f"*Confidence Score: {confidence_color} "
            f"{generated_answer.confidence_score:.2f}*"
        )
        
        # Display answer in a clean box
        st.markdown(
            f"""<div style="padding: 1.5rem; 
            border-radius: 0.5rem; 
            background-color: #f8f9fa; 
            border: 1px solid #dee2e6;
            color: #212529;
            margin: 1rem 0;
            line-height: 1.6;
            font-size: 1.1rem;">
            {generated_answer.answer}</div>""",
            unsafe_allow_html=True
        )
            
        # Display citations
        if generated_answer.citations:
            with st.expander("📚 Source Citations"):
                for i, citation in enumerate(generated_answer.citations, 1):
                    st.markdown(f"**Source {i}**: {Path(citation.source).name}")
                    st.markdown(f"*Relevance Score: {citation.score:.2f}*")
                    st.markdown(f'"{citation.text}"')
                    if citation.context:
                        st.markdown(f"*Context: {citation.context}*")
                    st.markdown("---")
        
        # Display timing information if available
        if generated_answer.metadata.get('generation_time'):
            st.markdown(
                f"*Answer generated in "
                f"{generated_answer.metadata['generation_time']:.2f} seconds*"
            )

    def _render_results_section(self, results: List[SearchResult]):
        """Render the detailed search results section"""
        st.markdown("### Detailed Search Results")
        
        # Display processing time if available
        if st.session_state.processing_time:
            st.markdown(
                f"*Search completed in {st.session_state.processing_time:.2f} seconds*"
            )
        
        # Display results in expandable sections
        for i, result in enumerate(results, 1):
            if hasattr(result, 'source'):
                with st.expander(
                    f"Result {i} - {Path(result.source).name} "
                    f"(Score: {result.score:.2f})"
                ):
                    st.markdown(result.text)
                    if result.context:
                        st.markdown("**Context:**")
                        st.markdown(f"*{result.context}*")
            else:
                st.warning("Result {i} is not in the expected format")

    def render_ui(self):
        """Render the main user interface"""
        st.title("AI-Powered Document Search")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # API key input if not set
            if not st.session_state.api_key:
                api_key = st.text_input(
                    "Enter Anthropic API Key",
                    type="password",
                    help="Required for document processing and search"
                )
                if api_key:
                    st.session_state.api_key = api_key
                    self._run_async(self._initialize_components())
            
            # Document upload section
            if st.session_state.components_ready:
                st.header("Document Processing")
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    accept_multiple_files=True,
                    type=['txt', 'pdf'],
                    help="Upload text or PDF documents for processing"
                )
                
                if uploaded_files:
                    if st.button("Process Documents", type="primary"):
                        self._run_async(self._process_documents(uploaded_files))
                
                # Show document statistics
                if st.session_state.documents:
                    st.markdown(
                        f"📚 **Processed Documents:** "
                        f"{len(st.session_state.documents)}"
                    )
                
                if st.session_state.processing_status:
                    st.info(st.session_state.processing_status)
        
        # Main search interface
        if st.session_state.components_ready:
            # Search input
            query = st.text_input(
                "Enter your search query",
                key="search_input",
                placeholder="What would you like to know about your documents?",
                help="Enter a question or search term"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                search_clicked = st.button("Search", type="primary")
            
            # Execute search
            if search_clicked and query:
                if not st.session_state.documents:
                    st.warning("Please upload and process some documents first.")
                else:
                    with st.spinner("Searching documents and generating answer..."):
                        search_output = self._run_async(self._perform_search(query))
                        if search_output:
                            results, generated_answer = search_output
                            st.session_state.search_results = (results, generated_answer)
                            st.session_state.last_query = query
            
            # Display results
            if st.session_state.search_results:
                results, generated_answer = st.session_state.search_results
                
                # Show the query that generated these results
                if st.session_state.last_query:
                    st.markdown(
                        f'*Results for: "{st.session_state.last_query}"*'
                    )
                
                # Render answer and results sections
                if generated_answer:
                    self._render_answer_section(generated_answer)
                self._render_results_section(results)
        
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
            # Cleanup resources
            self.executor.shutdown(wait=False)

if __name__ == "__main__":
    search_ui = StreamlitSearchUI()
    search_ui.run()