# new version
import os
from typing import Any, List

import asyncio
from dotenv import load_dotenv
from datetime import date
from dataclasses import dataclass

from groq import AsyncGroq

from pydantic import BaseModel, Field
from pydantic_ai.models.groq import GroqModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.result import RunResult

from tavily import AsyncTavilyClient

import streamlit as st

import nest_asyncio
nest_asyncio.apply()

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

    def __init__(
        self,
        max_results: int,
        todays_date: str,
    ):
        self.max_results = max_results
        self.todays_date = todays_date


class RAGResponse(BaseModel):
    research_title: str = Field(description='Markdown heading describing the article topic, prefixed with #')
    research_main: str = Field(description='The main content of the article in markdown with in-line citations')
    research_bullets: str = Field(description='A markdown list of bullet points summarizing the article with in-line citations')
    citations: List[str] = Field(description='List of searched sources')
    
@dataclass
class ResearchContext:
    citations: List[str]
    main_content: str


rag_agent = Agent(
    model=rag_model,
    result_type=RAGResponse,
    deps_type=SearchParameters,
)

@rag_agent.system_prompt
async def add_current_date(ctx: RunContext[SearchParameters]) -> str:
    todays_date = ctx.deps.todays_date
    system_prompt = ("""You are an exper research assistant answering questions about provided source materials.
    Your task is to:
    1. Search relevant information (1-5 searches)
    2. Use perform_search tool with query_number (1-5)
    3. Combine results within in-line citations [1], [2], etc for sources
    4. Provide key bullet points in markdown
    6. List citations with URLs

    Today's date: {{search_params.deps.todays_date}}

    Format:
    - Title (#)
    - Main content with [1], [2] in-linecitations
    - Key bullet points

    Example citation: "According to [1] source..." or "Research [2] shows that..." 
    """
    )
    return system_prompt

@rag_agent.tool
async def perform_search(search_params: RunContext[SearchParameters], query: str, query_number: int) -> Any:
    """Search for information using Tavily API."""
    search_results = await search_client.get_search_context(
        query=query,
        max_results=search_params.deps.max_results,
    )
    return search_results

## Interesting Question Model
# add an interesting question model
class InterestingQuestion(BaseModel):
    question: str = Field(description='An interesting question to ask a deeply curious and insightful research assistant')

# Define the prompt as a constant with word limit instruction
GENERATE_QUERY_PROMPT = ("""
    Think of an interesting and succinct one-sentence question to ask 
    to a deeply curious and insightful research assistant that is imbued 
    with the unending intellectual and creative resources that the world 
    wide web has to offer through search. Please ensure the question is 
    30 words or less.
    """
)

# Make interesting question agent
interesting_question_agent = Agent(
    rag_model,
    result_type=InterestingQuestion,
    system_prompt=GENERATE_QUERY_PROMPT
)

# Function to generate an interesting question with word limit
async def generate_interesting_question() -> str:
    """
    Generates an interesting research question based on the provided topic using the Groq LLaMA model.

    Returns:
        str: An interesting and succinct one-sentence research question limited to 30 words.
    """
    prompt = GENERATE_QUERY_PROMPT
    try:
        # Run the agent asynchronously and get the result
        response = await interesting_question_agent.run(prompt)
        
        # Access the data attribute of the RunResult
        question = response.data.question.strip()
        
        # Limit the question to 30 words
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


## Set up Streamlit UI
st.set_page_config(page_title="RAG-powered Search Assistant", layout="centered")

if 'research_results' not in st.session_state:
    st.session_state.research_results = None

st.title("RAG-powered Search Assistant")
st.write("Get AI-powered search summaries from the latest web content.")

# User input section
st.sidebar.title("Search Configuration")
if 'user_query' not in st.session_state:
    initial_question = asyncio.run(generate_interesting_question())
    st.session_state.user_query = initial_question

user_query = st.sidebar.text_area(
    label="Research Question:",
    value=st.session_state.get('user_query', "latest advancements in Artificial Intelligence"),
    help="This is the question that will guide your search.",
    height=150
)

results_per_search = st.sidebar.slider("Results per search:", min_value=1, max_value=5, value=3)


async def perform_rag_search(
    search_query: str, 
    max_search_results: int, 
    ) -> tuple[RunResult[RAGResponse], ResearchContext]:
    
    search_query = search_query[:400] if len(search_query) > 400 else search_query
    
    todays_date = date.today().strftime('%Y-%m-%d')
    
    search_params = SearchParameters(
        max_results=max_search_results, 
        todays_date=todays_date, 
    )
    
    result = await rag_agent.run(search_query, deps=search_params)
    
    research_context = ResearchContext(
        citations=[],
        main_content=result.data.research_main[:2000]
    )
    
    for i, url in enumerate(result.data.citations):
        research_context.citations.append(url)
    
    return result, research_context

if st.button("Start Research"):
    with st.spinner("Researching..."):
        try:
            rag_result, research_context = asyncio.run(
                perform_rag_search(
                    search_query=user_query,
                    max_search_results=results_per_search,
                )
            )
            st.session_state.research_context = research_context
            st.session_state.research_results = rag_result
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display research results if they exist
if st.session_state.research_results:
    rag_result = st.session_state.research_results
    try:
        if hasattr(rag_result, 'data'):
            st.markdown(rag_result.data.research_title)
            st.markdown(f"<div style='font-size: 16px;'>{rag_result.data.research_main}</div>", unsafe_allow_html=True)
            st.markdown(rag_result.data.research_bullets)
            
            st.markdown("### References")
            if rag_result.data.citations:
                for i, citation in enumerate(rag_result.data.citations, 1):
                    st.markdown(f"{i}. {citation}")
           
    except Exception as e:
        st.error(f"An error occurred: {e}")
