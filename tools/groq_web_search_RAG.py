# thanks to https://pub.towardsai.net/pydantic-ai-web-scraper-llama-3-3-python-powerful-ai-research-agent-6d634a9565fe for giving the starting point for this code

import os
import asyncio
from typing import Any, Dict, Optional, List, Literal
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
import re
from pydantic_ai.result import RunResult

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
    search_depth: Literal["basic", "advanced"] = "basic"
    time_range: str = "any"
    include_domains: List[str] = Field(default_factory=list)
    exclude_domains: List[str] = Field(default_factory=list)
    search_type: str = "general"
    language: str = "en"
    user_urls: List[str] = Field(default_factory=list)

    def __init__(
        self,
        max_results: int,
        todays_date: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        time_range: str = "any",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_type: str = "general",
        language: str = "en",
        user_urls: Optional[List[str]] = None
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
    # citations: List[str] = Field(description='List of citations with URLs for reference')
    user_provided_sources: List[str] = Field(description='List of user-provided sources')
    searched_sources: List[str] = Field(description='List of searched sources')

RAG_SYSTEM_PROMPT = """You are a research assistant. For each query:
1. Search relevant information (3-5 searches)
2. Use get_search tool with query_number (1-5)
3. Combine results within in-line citations [U1], [U2] for user provided sources and [S1], [S2] for searched sources.
4. Provide key bullet points
5. List user-provided citations with URLs
6. List searched citations with URLs

Today's date: {{search_params.deps.todays_date}}

Format:
- Title (#)
- Main content with [U1], [S2] in-linecitations
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
    """Search for information using Tavily API."""
    needed_results = min(search_params.deps.max_results, 3)
    
    # Convert None to empty lists for API call
    include_domains = search_params.deps.include_domains if search_params.deps.include_domains else []
    exclude_domains = search_params.deps.exclude_domains if search_params.deps.exclude_domains else []
    
    search_results = await search_client.get_search_context(
        query=query,
        max_results=needed_results,
        search_depth=search_params.deps.search_depth,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        search_type=search_params.deps.search_type,
        time_window=search_params.deps.time_range
    )
    
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

# Add a class to store the research context
@dataclass
class ResearchContext:
    sources: Dict[str, str]
    user_sources: List[str]
    searched_sources: List[str]
    main_content: str

# Initialize Q&A agent
qa_agent = Agent(
    rag_model,
    result_type=QAResponse,
    deps_type=ResearchContext,
    system_prompt=QA_SYSTEM_PROMPT
)

# add an interesting question model
class InterestingQuestion(BaseModel):
    question: str = Field(description='An interesting question to ask a deeply curious and insightful research assistant')

# Define the prompt as a constant with word limit instruction
GENERATE_QUERY_PROMPT = (
    """Think of an interesting and succinct one-sentence question to ask a deeply curious and insightful research assistant
   that is imbued with the unending intellectual and creative resources that the world wide web has to offer through search
   Please ensure the question is 30 words or less."""
)

# Make interesting question agent
interesting_question_agent = Agent(
    rag_model,
    result_type=InterestingQuestion,
    system_prompt=GENERATE_QUERY_PROMPT
)

# print(dir(interesting_question_agent))

# Function to generate an interesting question with word limit
async def generate_interesting_question(topic: str) -> str:
    """
    Generates an interesting research question based on the provided topic using the Groq LLaMA model.

    Args:
        topic (str): The research topic to frame the question around.

    Returns:
        str: An interesting and succinct one-sentence research question limited to 30 words.
    """
    prompt = GENERATE_QUERY_PROMPT
    try:
        # Run the agent synchronously and get the result
        response = interesting_question_agent.run_sync(prompt)
        
        # Access the data attribute of the RunResult
        question = response.data.question.strip()
        # print('question: ', question)
        
        # Limit the question to 30 words, not characters
        words = question.split()
        if len(words) > 30:
            question = ' '.join(words[:30]) + '...'
        
        return question
    except AttributeError:
        # Handle the case where 'complete' doesn't exist
        return "What are the latest advancements in Artificial Intelligence?"
    except Exception:
        # Silent failure with fallback question
        return "What are the latest advancements in Artificial Intelligence?"

# Set up Streamlit UI
st.set_page_config(page_title="RAG-powered Research Assistant", layout="centered")

st.title("RAG-powered Research Assistant")
st.write("Get AI-powered research summaries from the latest web content.")

# Initialize session state for user_query
if 'user_query' not in st.session_state:
    # Set a default topic for question generation
    default_topic = "latest developments in Large Language Models"
    # Automatically generate the research question without UI indications
    initial_question = asyncio.run(generate_interesting_question(default_topic))
    st.session_state.user_query = initial_question

# User input section
st.sidebar.title("Search Configuration")
user_query = st.sidebar.text_input(
    "Research Question:",
    value=st.session_state.get('user_query', "latest developments in Large Language Models"),
    help="This is the question that will guide your research."
)
results_per_search = st.sidebar.slider("Results per search:", min_value=1, max_value=5, value=3)

search_depth = st.sidebar.selectbox(
    "Search Depth:",
    options=["basic", "advanced"],
    help="Advanced searches take longer but may find more detailed information"
)

time_range = st.sidebar.selectbox(
    "Time Range to searchsince today:",
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
    help="Paste URLs or domains (e.g., arxiv.org, nature.com)",
    height=100
)

domains_exclude = st.sidebar.text_area(
    "Exclude Domains (one per line or comma-separated):",
    value="",
    help="Paste URLs or domains (e.g., reddit.com, nytimes.com)",
    height=100
)

# Clean and process the domains
include_domains = []
if domains_input.strip():
    raw_domains = [d.strip().lower() for d in re.split(r'[,\n]', domains_input) if d.strip()]
    for domain in raw_domains:
        domain = re.sub(r'^(https?://)?(www\.)?', '', domain).split('/')[0]
        if domain:
            include_domains.append(domain)

exclude_domains = []
if domains_exclude.strip():
    raw_domains = [d.strip().lower() for d in re.split(r'[,\n]', domains_exclude) if d.strip()]
    for domain in raw_domains:
        domain = re.sub(r'^(https?://)?(www\.)?', '', domain).split('/')[0]
        if domain:
            exclude_domains.append(domain)

# Process URLs from the text area
urls_input = st.sidebar.text_area(
    "Enter URLs to analyze (one per line or comma-separated):",
    value="",
    help="Paste full URLs to specific articles you want to analyze",
    height=100
)

# Clean and process the URLs which are  one per line or comma-separated
user_urls = []
if urls_input.strip():
    raw_urls = [url.strip() for url in re.split(r'[,\n]', urls_input) if url.strip()]
    for url in raw_urls:
        url = re.sub(r'^(https?://)?(www\.)?', '', url).split('/')[0]
        if url:
            user_urls.append(url)

# user_urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

st.write("Use the sidebar to configure your search parameters.")

# Store research results in session state
if 'research_context' not in st.session_state:
    st.session_state.research_context = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = None

# Modify the perform_rag_search function to handle context length
async def perform_rag_search(
    search_query: str, 
    max_search_results: int,
    search_depth: Literal["basic", "advanced"],
    time_range: str,
    search_type: str,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    user_urls: Optional[List[str]] = None
) -> tuple[RunResult[RAGResponse], ResearchContext]:
    # Truncate search query if too long
    search_query = search_query[:400] if len(search_query) > 400 else search_query
    
    search_params = SearchParameters(
        max_results=min(max_search_results, 5),
        todays_date=date.today().strftime("%Y-%m-%d"),
        search_depth=search_depth,
        time_range=time_range,
        search_type=search_type,
        include_domains=include_domains or [],
        exclude_domains=exclude_domains or [],
        user_urls=user_urls or []
    )
    
    rag_response = await rag_agent.run(search_query, deps=search_params)
    
    # Create research context
    context = ResearchContext(
        sources={},
        user_sources=[],
        searched_sources=[],
        main_content=rag_response.data.research_main[:2000]
    )
    
    # Populate sources
    for i, url in enumerate(rag_response.data.user_provided_sources):
        context.sources[f"U{i+1}"] = url
        context.user_sources.append(url)
    
    for i, url in enumerate(rag_response.data.searched_sources):
        context.sources[f"S{i+1}"] = url
        context.searched_sources.append(url)
    
    return rag_response, context

# Add Q&A function that uses the original research context with the existing user and searched sources with a new search if needed
async def answer_question(question: str, context: ResearchContext) -> RunResult[QAResponse]:
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
    
    If the answer is not clear or more information is needed, use the get_search tool to search again.
    """
    
    response = await qa_agent.run(context_prompt, deps=context)
    return response

# Update the "Start Research" button to use the generated user_query
if st.button("Start Research"):
    with st.spinner():
        try:
            rag_result, research_context = asyncio.run(
                perform_rag_search(
                    search_query=user_query,
                    max_search_results=results_per_search,
                    search_depth=search_depth,
                    time_range=time_range,
                    search_type=search_type,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    user_urls=user_urls
                )
            )
            st.session_state.research_context = research_context
            st.session_state.research_results = rag_result
        except Exception as e:
            st.error(f"Error during research: {str(e)}")

# Display research results if they exist
if st.session_state.research_results:
    rag_result = st.session_state.research_results
    try:
        if hasattr(rag_result, 'data'):
            st.markdown(rag_result.data.research_title)
            st.markdown(
                f"<div style='line-height:1.5;'>{rag_result.data.research_main}</div>", 
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
                with st.spinner():
                    if st.session_state.research_context is None:
                        st.error("No research context available")
                        st.stop()
                    
                    qa_response = asyncio.run(
                        answer_question(follow_up_question, st.session_state.research_context)
                    )
                    
                    # Access response data
                    if hasattr(qa_response, 'data'):
                        qa_container = st.container()
                        with qa_container:
                            st.markdown("### Answer")
                            st.markdown(qa_response.data.answer)
                            st.markdown("### Reasoning")
                            st.markdown(qa_response.data.reasoning)
                            st.markdown("### Sources Used")
                            
                            # Extract citation numbers from the answer and reasoning text
                            answer_text = f"{qa_response.data.answer} {qa_response.data.reasoning}"
                            
                            # Find all citations in the format [U1], [S2], [N1], etc.
                            citations = re.findall(r'\[(U|S|N)\d+\]', answer_text)
                            
                            # Only proceed if research context exists and has sources
                            if (st.session_state.research_context is not None and 
                                hasattr(st.session_state.research_context, 'sources')):
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
                                        url = st.session_state.research_context.sources.get(key, "URL Not Found")
                                        st.markdown(f"{citation}: {url}")
                                
                                if searched_citations:
                                    st.markdown("#### Original Search Sources")
                                    for citation in sorted(set(searched_citations)):
                                        idx = citation[2:-1]
                                        key = f"S{idx}"
                                        url = st.session_state.research_context.sources.get(key, "URL Not Found")
                                        st.markdown(f"{citation}: {url}")
                                
                                if new_citations:
                                    st.markdown("#### Newly Found Sources")
                                    for citation in sorted(set(new_citations)):
                                        idx = citation[2:-1]
                                        key = f"N{idx}"
                                        if key not in st.session_state.research_context.sources:
                                            st.session_state.research_context.sources[key] = "New Source URL Placeholder"
                                            st.session_state.research_context.searched_sources.append(key)
    except Exception as e:
        st.error(f"Error processing research results: {str(e)}")
