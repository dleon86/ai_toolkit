import os
import asyncio
from typing import Any, Dict, Optional
from datetime import date
from dataclasses import dataclass

import nest_asyncio
nest_asyncio.apply()

from groq import AsyncGroq
from pydantic_ai.models.groq import GroqModel
import streamlit as st
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Initialize API clients
groq_client = AsyncGroq(
    api_key=os.getenv('GROQ_API_KEY'),
)

# Initialize Groq LLM model for RAG
rag_model = GroqModel('llama3-groq-70b-8192-tool-use-preview')

# Initialize Tavily search client
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not TAVILY_API_KEY:
    raise ValueError('TAVILY_API_KEY environment variable is not set')

search_client = AsyncTavilyClient(TAVILY_API_KEY)

@dataclass
class SearchParameters:
    max_results: int
    todays_date: str
    search_depth: str
    time_range: str
    include_domains: list[str]
    exclude_domains: list[str]
    search_type: str
    language: str
    user_urls: List[str]

    def __init__(
        self,
        max_results: int,
        todays_date: str,
        search_depth: str = "basic",
        time_range: str = "any",
        include_domains: list[str] = None,
        exclude_domains: list[str] = None,
        search_type: str = "general",
        language: str = "en",
        user_urls: List[str] = None
    ):
        self.max_results = max_results
        self.todays_date = todays_date
        self.search_depth = search_depth
        self.time_range = time_range
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []
        self.search_type = search_type
        self.language = language
        self.user_urls = user_urls or []

class RAGResponse(BaseModel):
    research_title: str = Field(description='Markdown heading describing the article topic, prefixed with #')
    research_main: str = Field(description='The main content of the article with inline citations')
    research_bullets: str = Field(description='A list of bullet points summarizing the article')
    citations: List[str] = Field(description='List of citations with URLs for reference')
    user_provided_sources: List[str] = Field(description='List of user-provided sources')
    searched_sources: List[str] = Field(description='List of searched sources')

RAG_SYSTEM_PROMPT = """You are a research assistant. For each query:
1. Search relevant information (3-5 searches)
2. Use get_search tool with query_number (1-3)
3. Combine results with citations [1], [2], etc.
4. Provide key bullet points
5. List citations with URLs

Today's date: {{search_params.deps.todays_date}}

Format:
- Title (#)
- Main content with [1], [2] citations
- Key bullet points
- Numbered citations with URLs"""

rag_agent = Agent(
    rag_model,
    result_type=RAGResponse,
    deps_type=SearchParameters,
    system_prompt=RAG_SYSTEM_PROMPT
)

@rag_agent.tool
async def get_search(search_params: RunContext[SearchParameters], query: str, query_number: int) -> Any:
    """Search for information using Tavily API.
    
    Args:
        query: The search query string
        query_number: Sequential number of this search (1, 2, or 3)
    Returns:
        The search results from Tavily
    """
    # Calculate how many additional results we need from search
    # If we have user URLs, we'll search for 2 additional sources
    user_url_count = len(search_params.deps.user_urls)
    needed_results = 2 if user_url_count > 0 else search_params.deps.max_results
    
    # Convert empty lists to None to avoid serialization issues
    include_domains = search_params.deps.include_domains if search_params.deps.include_domains else None
    exclude_domains = search_params.deps.exclude_domains if search_params.deps.exclude_domains else None
    
    search_results = await search_client.get_search_context(
        query=query,
        max_results=needed_results,
        search_depth=search_params.deps.search_depth,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        search_type=search_params.deps.search_type,
        time_window=search_params.deps.time_range
    )
    
    # Truncate each search result content to manage context length
    for result in search_results:
        if 'content' in result:
            result['content'] = result['content'][:1000]  # Limit content length
    
    return search_results

# Add new model for Q&A responses
class QAResponse(BaseModel):
    answer: str = Field(description='Direct answer to the user question')
    reasoning: str = Field(description='Explanation of how the answer was derived')
    sources_used: List[str] = Field(description='List of sources used to answer the question')

# Add Q&A system prompt
QA_SYSTEM_PROMPT = """You are a research assistant answering questions about provided source materials.
Your task is to:
1. Answer the user's question based on the provided sources
2. Explain your reasoning
3. Cite specific sources used to form your answer
4. Distinguish between user-provided sources and searched sources in your citations

Format your response with:
- A direct answer to the question
- Your reasoning process
- Citations to specific sources used, marking user-provided sources with [U] and searched sources with [S]

Example citation: "According to [U1] user's first source..." or "Research shows [S2] from the second searched source..."
"""

# Initialize Q&A agent
qa_agent = Agent(
    rag_model,
    result_type=QAResponse,
    system_prompt=QA_SYSTEM_PROMPT
)

# Add a class to store the research context
@dataclass
class ResearchContext:
    sources: Dict[str, str]
    user_sources: List[str]
    searched_sources: List[str]
    main_content: str

# Update perform_rag_search to better manage context
async def perform_rag_search(
    search_query: str, 
    max_search_results: int,
    user_urls: Optional[List[str]] = None
) -> tuple[RAGResponse, ResearchContext]:
    current_date = date.today()
    date_string = current_date.strftime("%Y-%m-%d")
    
    # Truncate search query if too long
    search_query = search_query[:200] if len(search_query) > 200 else search_query
    
    search_params = SearchParameters(
        max_results=min(max_search_results, 3),  # Limit max results
        todays_date=date_string,
        search_depth=search_depth,
        time_range=time_range,
        search_type=search_type,
        include_domains=include_domains if include_domains else [],
        exclude_domains=[],
        user_urls=user_urls[:3] if user_urls else []  # Limit user URLs
    )
    
    rag_response = await rag_agent.run(search_query, deps=search_params)
    
    # Create research context
    context = ResearchContext(
        sources={},
        user_sources=[],
        searched_sources=[],
        main_content=rag_response.data.research_main[:2000]  # Limit main content length
    )
    
    # Populate sources (limited number)
    for i, url in enumerate(rag_response.data.user_provided_sources[:3]):
        context.sources[f"U{i+1}"] = url
        context.user_sources.append(url)
    
    for i, url in enumerate(rag_response.data.searched_sources[:3]):
        context.sources[f"S{i+1}"] = url
        context.searched_sources.append(url)
    
    return rag_response, context

# Add Q&A function
async def answer_question(question: str, context: ResearchContext) -> QAResponse:
    # Create a prompt that includes the context
    context_prompt = f"""
    Question: {question}
    
    Available sources:
    User-provided sources:
    {chr(10).join(f'[U{i+1}] {url}' for i, url in enumerate(context.user_sources))}
    
    Searched sources:
    {chr(10).join(f'[S{i+1}] {url}' for i, url in enumerate(context.searched_sources))}
    
    Content:
    {context.main_content}
    """
    
    response = await qa_agent.run(context_prompt)
    return response

# Set up streamlit UI
st.set_page_config(page_title="RAG-powered Research Assistant", layout="centered")

st.title("RAG-powered Research Assistant")
st.write("Get AI-powered research summaries from the latest web content.")

# User input section
st.sidebar.title("Search Configuration")
user_query = st.sidebar.text_input("Enter your research query:", value="latest developments in Large Language Models")
results_per_search = st.sidebar.slider("Results per search:", min_value=1, max_value=5, value=3)

search_depth = st.sidebar.selectbox(
    "Search Depth:",
    options=["basic", "advanced"],
    help="Advanced searches take longer but may find more detailed information"
)

time_range = st.sidebar.selectbox(
    "Time Range:",
    options=["any", "day", "week", "month", "year"],
    help="Filter results by recency"
)

search_type = st.sidebar.selectbox(
    "Search Type:",
    options=["general", "news", "academic", "technical", "blogs", "reddit", "youtube"],
    help="Focus on specific types of content"
)

domains_input = st.sidebar.text_area(
    "Include Domains (one per line or comma-separated):",
    value="",
    help="Paste URLs or domains (e.g., arxiv.org, https://nature.com)",
    height=100
)

# Clean and process the domains
if domains_input.strip():
    # Split on both commas and newlines
    raw_domains = [d for d in domains_input.replace(',', '\n').split('\n') if d.strip()]
    
    # Clean up the domains (remove http(s):// and www., and extract domain)
    include_domains = []
    for domain in raw_domains:
        domain = domain.strip().lower()
        # Remove http(s):// and www.
        domain = domain.replace('https://', '').replace('http://', '').replace('www.', '')
        # Remove any paths or query parameters (keep only domain)
        domain = domain.split('/')[0]
        if domain:
            include_domains.append(domain)
else:
    include_domains = []

# Process URLs from the text area
urls_input = st.sidebar.text_area(
    "Enter URLs to analyze (one per line):",
    value="",
    help="Paste full URLs to specific articles you want to analyze",
    height=100
)

# Clean and process the URLs
user_urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

st.write("Use the sidebar to configure your search parameters.")

# Store research results in session state
if 'research_context' not in st.session_state:
    st.session_state.research_context = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = None

if st.button("Start Research"):
    with st.spinner("Researching your query..."):
        rag_result, research_context = asyncio.run(
            perform_rag_search(
                search_query=user_query,
                max_search_results=results_per_search,
                user_urls=user_urls
            )
        )
        st.session_state.research_context = research_context
        st.session_state.research_results = rag_result

# Display research results if they exist
if st.session_state.research_results:
    rag_result = st.session_state.research_results
    try:
        if hasattr(rag_result, 'data'):
            st.markdown(rag_result.data.research_title)
            st.markdown(
                f"<div style='line-height:1.6;'>{rag_result.data.research_main}</div>", 
                unsafe_allow_html=True
            )
            st.markdown("### Key Takeaways")
            st.markdown(rag_result.data.research_bullets)
            
            st.markdown("### References")
            if rag_result.data.user_provided_sources:
                st.markdown("#### User-Provided Sources")
                for i, citation in enumerate(rag_result.data.user_provided_sources, 1):
                    st.markdown(f"U{i}. {citation}")
            
            if rag_result.data.searched_sources:
                st.markdown("#### Additional Searched Sources")
                for i, citation in enumerate(rag_result.data.searched_sources, 1):
                    st.markdown(f"S{i}. {citation}")

            # Add Q&A section only after research is done
            st.markdown("---")
            st.markdown("### Ask Follow-up Questions")
            follow_up_question = st.text_input("Ask a question about the research results:", "")
            
            if follow_up_question and st.button("Ask Question"):
                with st.spinner("Analyzing sources and searching if needed..."):
                    qa_response = asyncio.run(
                        answer_question(follow_up_question, st.session_state.research_context)
                    )
                    
                    qa_container = st.container()
                    with qa_container:
                        st.markdown("### Answer")
                        st.markdown(qa_response.data.answer)
                        st.markdown("### Reasoning")
                        st.markdown(qa_response.data.reasoning)
                        st.markdown("### Sources Used")
                        
                        # Extract citation numbers from the answer and reasoning text
                        answer_text = qa_response.data.answer + " " + qa_response.data.reasoning
                        
                        # Find all citations in the format [U1], [S2], [N1], etc.
                        import re
                        citations = re.findall(r'\[(U|S|N)\d+\]', answer_text)
                        
                        # Group cited sources by type
                        user_citations = [c for c in citations if c.startswith('[U')]
                        searched_citations = [c for c in citations if c.startswith('[S')]
                        new_citations = [c for c in citations if c.startswith('[N')]
                        
                        if user_citations:
                            st.markdown("#### User-Provided Sources")
                            for citation in sorted(set(user_citations)):
                                # Extract the index number from citation (e.g., "U1" from "[U1]")
                                idx = citation[2:-1]  # Remove [ ] and U/S/N
                                key = f"U{idx}"
                                url = st.session_state.research_context.sources[key]
                                st.markdown(f"{citation}: {url}")
                        
                        if searched_citations:
                            st.markdown("#### Original Search Sources")
                            for citation in sorted(set(searched_citations)):
                                idx = citation[2:-1]
                                key = f"S{idx}"
                                url = st.session_state.research_context.sources[key]
                                st.markdown(f"{citation}: {url}")
                        
                        if new_citations:
                            st.markdown("#### Newly Found Sources")
                            for citation in sorted(set(new_citations)):
                                idx = citation[2:-1]
                                key = f"N{idx}"
                                url = st.session_state.research_context.sources[key]
                                st.markdown(f"{citation}: {url}")
                            
                            # Update the research context with new sources
                            if new_citations and new_citations not in st.session_state.research_context.searched_sources:
                                st.session_state.research_context.searched_sources.extend(new_citations)
    except Exception as e:
        st.error(f"Error processing research results: {str(e)}") 