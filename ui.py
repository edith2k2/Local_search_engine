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
from retriever import ChainOfThoughtRetriever, SearchResult, SearchIteration
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
            page_title="Local Search Engine",
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
            'max_search_steps': 1,
            'top_k': 5,
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
            'processing_time': None,
            'current_iterations': None 
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
    
    def render_temporal_controls(self, prefix: str = "") -> Optional[TemporalConstraints]:
        """
        Render temporal search controls
        Args:
            prefix: String to prefix to keys to make them unique
        """
        time_frame = st.radio(
            "Select time frame:",
            ["All Time", "Custom Range", "Recent"],
            key=f"{prefix}time_frame_radio"
        )
        
        if time_frame == "Custom Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", key=f"{prefix}start_date_input")
            with col2:
                end_date = st.date_input("End Date", key=f"{prefix}end_date_input")
                
            strict_temporal = st.checkbox(
                "Strict time boundaries",
                help="If checked, only show results exactly within the time range",
                key=f"{prefix}strict_temporal_checkbox"
            )
            
            return TemporalConstraints(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
                time_frame=TimeFrame.STRICT if strict_temporal else TimeFrame.FLEXIBLE
            )
            
        elif time_frame == "Recent":
            period = st.selectbox(
                "Time period:",
                ["Last 24 Hours", "Last Week", "Last Month"],
                key=f"{prefix}recent_period_select"
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
            
        return None 
    
    def _render_results_section(self, results: List[SearchResult]):
        """Render the detailed search results section"""
        # st.markdown("### Detailed Search Results")
        
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

    def _render_chain_of_thought_section(self, iterations: List[SearchIteration]):
        """
        Renders the chain of thought process using tabs and containers instead of nested expanders.
        This approach maintains organization while avoiding the nested expander limitation.
        """
        with st.expander(" View Search Reasoning Process", expanded=False):
            # Introduction text
            # st.markdown("""
            #     <div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; margin-bottom: 1rem;'>
            #     This section shows how the search engine refined its understanding of your query through multiple steps. 
            #     Each step includes the reasoning process and any query refinements made to improve results.
            #     </div>
            #     """, unsafe_allow_html=True)
            
            # Create tabs for each iteration
            if len(iterations) > 0:
                tabs = st.tabs([f"Step {i+1}" for i in range(len(iterations))])
                
                for i, (tab, iteration) in enumerate(zip(tabs, iterations)):
                    with tab:
                        # Display basic iteration information
                        st.markdown(f"**Current Query:** {iteration.query}")
                        
                        if iteration.reasoning:
                            # Create sections using columns and containers instead of expanders
                            st.markdown("#### Analysis Summary")
                            
                            # Display reasoning in a styled container
                            st.markdown("**Detailed Analysis:**")
                            st.markdown(
                                f"""<div style="padding: 1.5rem; 
                                border-left: 3px solid #3498db; 
                                background-color: #343a40; 
                                color: #f8f9fa;
                                border-radius: 0.5rem;
                                margin: 0.75rem 0;
                                line-height: 1.6;
                                font-size: 1.1rem;">
                                {iteration.reasoning.reasoning_explanation}
                                </div>""",
                                unsafe_allow_html=True
                            )
                            
                            # Create two columns for gaps and refinements
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if iteration.reasoning.gaps_identified:
                                    st.markdown("#### 🔍 Identified Gaps")
                                    for gap in iteration.reasoning.gaps_identified:
                                        st.markdown(f"• {gap}")
                            
                            with col2:
                                if iteration.reasoning.suggested_refinement:
                                    st.markdown("#### 🔄 Query Refinement")
                                    st.markdown(f"**Refined to:** {iteration.reasoning.suggested_refinement}")
                                    st.markdown("**Reason for refinement:**")
                                    st.markdown(iteration.reasoning.reasoning_explanation.split('\n')[0])
                            
                            # Show relevance scores in a table
                            # if iteration.reasoning.relevance_findings:
                            #     st.markdown("#### 📈 Relevance Scores")
                            #     scores_df = pd.DataFrame(
                            #         [(f"Result {k}", f"{v:.2f}") for k, v in iteration.reasoning.relevance_findings.items()],
                            #         columns=["Result", "Score"]
                            #     )
                            #     st.dataframe(
                            #         scores_df,
                            #         hide_index=True,
                            #         use_container_width=True
                            #     )

    async def _perform_search_with_progress(self, query: str, temporal_constraints: Optional[TemporalConstraints] = None):
        """
        Executes the search and answer generation with progressive display:
        1. Answer container at top
        2. Steps in middle
        3. Sources at bottom
        """
        try:
            # Create all containers in visual order, but use regular containers
            answer_container = st.container()  # Will be filled last but appears at top
            answer_status = st.empty()  # Status placeholder inside answer section
            
            divider1 = st.container()  # Visual divider
            
            steps_container = st.container()  # Middle
            sources_container = st.container()  # Bottom
            
            search_status = st.empty()  # Search status indicator

            # Start search process
            search_status.info("🔍 Performing search...")
            start_time = time.time()

            # Create search parameters with temporal constraints
            search_params = SearchParameters(
                query=query,
                ui_temporal=temporal_constraints,
                nl_temporal=None,
                strict_temporal=True if temporal_constraints and temporal_constraints.has_constraints else False
            )
            # Execute search with steps
            # results, iterations = await st.session_state.retriever.search(
            #     query, 
            #     return_steps=True
            # )
            results, iterations = await st.session_state.retriever.search_with_parameters(
                params=search_params,
                return_steps=True
            )

            # Store iterations for persistence
            st.session_state.current_iterations = iterations

            # Clear search status
            search_status.empty()

            # Immediately display search steps if available
            with steps_container:
                if iterations:
                    st.markdown("### Search Process and Analysis")
                    self._render_chain_of_thought_section(iterations)

            # Immediately display source documents
            with sources_container:
                if results:
                    st.markdown("### Source Documents")
                    st.markdown(f"*Search completed in {time.time() - start_time:.2f} seconds*")
                    self._render_results_section(results)

            # Show answer generation status in the answer section
            answer_status.info("Generating comprehensive answer...")

            # Generate answer
            generated_answer = await st.session_state.answer_generator.generate_answer(
                query,
                results
            )

            # Clear answer status
            answer_status.empty()

            # Display answer at the top
            with answer_container:
                if generated_answer:
                    if st.session_state.last_query:
                        st.markdown(f'*Results for: "{st.session_state.last_query}"*')
                    self._render_answer_section(generated_answer)
                    
                    # Add visual separator
                    with divider1:
                        st.markdown("---")

            # Store final results in session state
            st.session_state.search_results = (results, generated_answer)
            st.session_state.last_query = query

        except Exception as e:
            st.error(f"Error during search: {str(e)}")

    def format_time_range(self, temporal_constraints):
        """Convert datetime range to human readable format"""
        if not temporal_constraints.start_date:
            return "from any time until now"
            
        now = datetime.now()
        start = temporal_constraints.start_date
        end = temporal_constraints.end_date or now
        
        # Calculate the difference
        diff_start = now - start
        
        # Convert to natural language
        if diff_start.days == 0:  # Today
            if end == now:
                return "from today until now"
            return f"from today until {end.strftime('%B %d')}"
        elif diff_start.days == 1:  # Yesterday
            return "from yesterday until now"
        elif diff_start.days < 7:  # Last few days
            return f"from {diff_start.days} days ago until now"
        elif diff_start.days < 31:  # Weeks
            weeks = diff_start.days // 7
            return f"from {weeks} week{'s' if weeks > 1 else ''} ago until now"
        else:  # Months/specific dates
            return f"from {start.strftime('%B %d, %Y')} to {end.strftime('%B %d, %Y')}"
        
    def render_ui(self):
        """Render the main user interface"""
        st.title("AI-Powered Document Search")
        
        # temporal_constraints = self.render_temporal_controls()
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Time Range section in collapsible expander
            with st.expander("🕒 Time Range", expanded=False):
                temporal_constraints = self.render_temporal_controls(prefix="sidebar_")
            
            # Search Configuration in collapsible expander
            with st.expander("⚙️ Search Configuration", expanded=False):
                st.number_input(
                    "Maximum chain of thought search steps",
                    min_value=0,
                    max_value=5,
                    value=st.session_state.max_search_steps,
                    key="steps_input",
                    help="Set the maximum number iterations for chain of thought search",
                    on_change=self._on_steps_change
                )
                top_k = st.number_input(
                    "Number of results to retrieve (top-k)",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.top_k,
                    step=1,
                    help="Set the number of top results to retrieve from search"
                )

                if top_k != st.session_state.top_k:
                    st.session_state.top_k = top_k
                    if st.session_state.retriever:
                        st.session_state.retriever.top_k = top_k

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
            
            # Document upload section in collapsible expander
            if st.session_state.components_ready:
                with st.expander("📁 Document Upload", expanded=False):
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
                    
                    if st.session_state.processing_status:
                        st.info(st.session_state.processing_status)

            # Separate document list expander
            if st.session_state.components_ready and st.session_state.documents:
                with st.expander(f"📚 Document Library ({len(st.session_state.documents)})", expanded=False):
                    # Display document count
                    st.markdown(f"**Total Documents:** {len(st.session_state.documents)}")
                    st.markdown("---")
                    
                    # Display document details in a scrollable container
                    for file_path, doc_data in st.session_state.documents.items():
                        metadata = doc_data['metadata']
                        
                        # Create a clean, compact display for each document
                        st.markdown(
                            f"""<div style='padding: 5px 0;'>
                            <strong>{metadata['file_name']}</strong><br/>
                            <small>
                            📄 {metadata['file_type']} | 
                            💾 {format_size(metadata['size_bytes'])} | 
                            📑 {metadata['num_chunks']} chunks<br/>
                            📅 Added: {format_date(metadata['created_time'])}
                            </small>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                        st.markdown("---")
        
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
                st.info("🕒 Search " + self.format_time_range(temporal_constraints)
)

            col1, col2 = st.columns([1, 4])
            with col1:
                search_clicked = st.button("Search", type="primary")
            
            # Handle search
            if search_clicked and query:
                if not st.session_state.documents:
                    st.warning("Please upload and process some documents first.")
                else:
                    # Clear existing results
                    st.session_state.search_results = None
                    asyncio.run(self._perform_search_with_progress(query, temporal_constraints))
            
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