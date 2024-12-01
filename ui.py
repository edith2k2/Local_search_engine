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
from datetime import datetime, timedelta
import logging

# Import our search components
from preprocessing import AsyncDocumentProcessor, BatchConfig
from retriever import ChainOfThoughtRetriever, SearchResult
from answer_generator import AnswerGenerator, GeneratedAnswer
from query_parser import TemporalConstraints, TimeFrame, SearchParameters

# Enable nested async support for Streamlit
nest_asyncio.apply()
logger = logging.getLogger(__name__)

# Helper functions to format the display
def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"

def format_date(date_string: str) -> str:
    """Format ISO date string to readable format"""
    try:
        date = datetime.fromisoformat(date_string)
        return date.strftime("%Y-%m-%d %H:%M")
    except:
        return date_string
    
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
            'max_search_steps': 3,
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

    def _update_search_steps(self, new_value: int):
        """Update max search steps with validation"""
        if 1 <= new_value <= 5:
            st.session_state.max_search_steps = new_value
            # Optionally update retriever configuration
            if st.session_state.retriever:
                logger.info(f"Updating max search steps to {new_value}")
                st.session_state.retriever.max_iterations = new_value

    def _on_steps_change(self):
        """Callback for when max search steps changes"""
        try:
            new_value = st.session_state.steps_input
            if 0 <= new_value <= 5:
                st.session_state.max_search_steps = new_value
                if st.session_state.retriever:
                    logger.info(f"Updating max search steps to {new_value}")
                    st.session_state.retriever.max_iterations = new_value
                    # Add visual feedback
                    st.success(f"Search steps updated to {new_value}")
            else:
                st.error("Search steps must be between 1 and 5")
        except Exception as e:
            logger.error(f"Error updating search steps: {str(e)}")
            st.error("Failed to update search steps")

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
                max_iterations=st.session_state['max_search_steps'],
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
            new_documents = await st.session_state.processor.process_documents(file_paths)
            
            if new_documents:
                # Update our session state with the new documents
                st.session_state.documents.update(new_documents)
                
                # Save the updated indices
                await st.session_state.processor.save_indices(str(self.config['index_directory']))
                
                # Update the retriever with the complete set of documents
                if st.session_state.retriever:
                    st.session_state.retriever.update_documents(st.session_state.documents)
                
                # Update status with meaningful information
                total_chunks = sum(
                    doc_data['metadata']['num_chunks'] 
                    for doc_data in new_documents.values()
                )
                
                st.session_state.processing_status = (
                    f"Successfully processed {len(new_documents)} new documents "
                    f"containing {total_chunks} total chunks. "
                    f"Total documents in system: {len(st.session_state.documents)}"
                )
            else:
                st.session_state.error_message = "No documents were successfully processed"
            
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
        query: str,
        temporal_constraints: Optional[TemporalConstraints] = None
    ) -> Optional[Tuple[List[SearchResult], GeneratedAnswer]]:
        """Execute search and generate answer with temporal awareness"""
        try:
            # Get search results
            start_time = time.time()
            
            # If we have temporal constraints, apply them
            if temporal_constraints and temporal_constraints.has_constraints:
                search_params = SearchParameters(
                    query=query,
                    # If we have UI-specified temporal constraints, use them
                    ui_temporal=temporal_constraints,
                    # Natural language temporal parsing will happen in the retriever
                    nl_temporal=None,
                    # Default to strict temporal matching if constraints are provided
                    strict_temporal=True if temporal_constraints and temporal_constraints.has_constraints else False
                )
                results, retrieval_steps = await st.session_state.retriever.search_with_parameters(
                    params=search_params,
                    return_steps=True
                )
            else:
                results, retrieval_steps = await st.session_state.retriever.search(query)
            
            if not results:
                return None
            
            # Generate answer from results
            generated_answer = await st.session_state.answer_generator.generate_answer(
                query,
                results
            )
            
            # Add temporal information to the answer metadata
            if temporal_constraints:
                generated_answer.metadata['temporal_constraints'] = {
                    'start_date': temporal_constraints.start_date.isoformat() if temporal_constraints.start_date else None,
                    'end_date': temporal_constraints.end_date.isoformat() if temporal_constraints.end_date else None,
                    'time_frame': temporal_constraints.time_frame.value
                }
            
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
        # st.markdown(
        #     f"*Confidence Score: {confidence_color} "
        #     f"{generated_answer.confidence_score:.2f}*"
        # )
        
        # Display answer in a clean box
        st.markdown(
            f"""<div style="padding: 1.5rem; 
            border-radius: 0.5rem; 
            background-color: #343a40; 
            border: 1px solid #495057;
            color: #f8f9fa;
            margin: 1rem 0;
            line-height: 1.6;
            font-size: 1.1rem;">
            {generated_answer.answer}</div>""",
            unsafe_allow_html=True
        )
            
        # Display citations
        # if generated_answer.citations:
        #     with st.expander("📚 Source Citations"):
        #         for i, citation in enumerate(generated_answer.citations, 1):
        #             st.markdown(f"**Source {i}**: {Path(citation.source).name}")
        #             st.markdown(f"*Relevance Score: {citation.score:.2f}*")
        #             st.markdown(f'"{citation.text}"')
        #             if citation.context:
        #                 st.markdown(f"*Context: {citation.context}*")
        #             st.markdown("---")
        
        # Display timing information if available
        if generated_answer.metadata.get('generation_time'):
            st.markdown(
                f"*Answer generated in "
                f"{generated_answer.metadata['generation_time']:.2f} seconds*"
            )
    
    def render_temporal_controls(self) -> Optional[TemporalConstraints]:
        """Render temporal search controls in the sidebar"""
        st.sidebar.header("Time Range")
        
        time_frame = st.sidebar.radio(
            "Select time frame:",
            ["All Time", "Custom Range", "Recent"]
        )
        
        if time_frame == "Custom Range":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
                
            strict_temporal = st.sidebar.checkbox(
                "Strict time boundaries",
                help="If checked, only show results exactly within the time range"
            )
            
            return TemporalConstraints(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
                time_frame=TimeFrame.STRICT if strict_temporal else TimeFrame.FLEXIBLE
            )
            
        elif time_frame == "Recent":
            period = st.sidebar.selectbox(
                "Time period:",
                ["Last 24 Hours", "Last Week", "Last Month"]
            )
            
            now = datetime.now()
            if period == "Last 24 Hours":
                start_date = now - timedelta(days=1)
            elif period == "Last Week":
                start_date = now - timedelta(weeks=1)
            else:
                start_date = now - timedelta(days=30)
                
            return TemporalConstraints(
                start_date=start_date,
                end_date=now,
                time_frame=TimeFrame.FLEXIBLE
            )
            
        return None  # All Time selected
    
    def _render_results_section(self, results: List[SearchResult]):
        """Render the detailed search results section"""
        st.markdown("### Detailed Search Results")
        
        # Display processing time if available
        if st.session_state.processing_time:
            st.markdown(
                f"*chain of thought search with {st.session_state.retriever.max_iterations} iterations completed in {st.session_state.processing_time:.2f} seconds*"
            )
        
        # Group results by source document
        results_by_source = {}
        for result in results:
            if result.source not in results_by_source:
                results_by_source[result.source] = []
            results_by_source[result.source].append(result)
        
        for source, source_results in results_by_source.items():
            with st.expander(f"📄 {Path(source).name} ({len(source_results)} results)"):
                for i, result in enumerate(source_results, 1):
                    # Add document date if available
                    doc_date = datetime.fromisoformat(
                        result.metadata.get('created_time', '')
                    ).strftime('%Y-%m-%d %H:%M:%S')
                    
                    st.markdown(f"**Document Date:** {doc_date}")
                    
                    # Show temporal score if available
                    if 'temporal_score' in result.metadata:
                        st.markdown(
                            f"*Temporal Relevance: {result.metadata['temporal_score']:.2f}*"
                        )

                    st.markdown(
                        f"""<div style="
                        padding: 1rem;
                        border-left: 3px solid #3498db;
                        background-color: #343a40;
                        color: #f8f9fa;
                        margin: 0.5rem 0;">
                        <p>{result.text}</p>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    
                    if result.context:
                        st.markdown("**Context:**")
                        st.markdown(f"*{result.context}*")
                    
                    if result.reasoning:
                        st.markdown("**Analysis:**")
                        st.markdown(f"*{result.reasoning}*")
                    
                    if i < len(source_results):
                        st.markdown("---")

    def render_ui(self):
        """Render the main user interface"""
        st.title("AI-Powered Document Search")
        
        temporal_constraints = self.render_temporal_controls()
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            st.header("Search Configuration")
            
            # Use number_input for better control
            st.number_input(
                "Maximum chain of thought search steps",
                min_value=0,
                max_value=5,
                value=st.session_state.max_search_steps,
                key="steps_input",
                help="Set the maximum number iterations for chain of thought search",
                on_change=self._on_steps_change
            )

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
                with st.form("document_upload_form"):
                    uploaded_files = st.file_uploader(
                        "Upload Documents",
                        accept_multiple_files=True,
                        type=['txt', 'pdf'],
                        help="Upload text or PDF documents for processing"
                    )
                    
                    submit_button = st.form_submit_button("Process Documents")
                    
                    if submit_button and uploaded_files:
                        with st.spinner("Processing documents..."):
                            self._run_async(self._process_documents(uploaded_files))
                
                # if uploaded_files:
                #     if st.button("Process Documents", type="primary"):
                #         self._run_async(self._process_documents(uploaded_files))
                
                # Show document statistics
                if st.session_state.documents:
                    st.markdown(
                        f"📚 **Processed Documents:** "
                        f"{len(st.session_state.documents)}"
                    )
                    # st.sidebar.header("📚 Document Library")
    
                    # Create an expander to show document details
                    with st.sidebar.expander(f"View All Documents ({len(st.session_state.documents)})", expanded=False):
                        # Iterate through documents and display their information
                        for file_path, doc_data in st.session_state.documents.items():
                            metadata = doc_data['metadata']
                            
                            # Create a clean display for each document
                            st.markdown(f"""
                            **{metadata['file_name']}**  
                            - Size: {format_size(metadata['size_bytes'])}
                            - Chunks: {metadata['num_chunks']}
                            - Type: {metadata['file_type']}
                            - Added: {format_date(metadata['created_time'])}
                            ---
            """)
                
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
            
            if temporal_constraints and temporal_constraints.has_constraints:
                st.info(
                    "🕒 Searching within time range: "
                    f"{temporal_constraints.start_date.strftime('%Y-%m-%d %H:%M:%S') if temporal_constraints.start_date else 'any time'} "
                    f"to {temporal_constraints.end_date.strftime('%Y-%m-%d %H:%M:%S') if temporal_constraints.end_date else 'now'}"
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
                        search_output = self._run_async(self._perform_search(query, temporal_constraints))
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