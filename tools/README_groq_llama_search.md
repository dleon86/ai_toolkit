# Groq Llama 3 Hybrid Search & Q&A

A powerful research tool that combines user-provided sources with web search results, powered by Groq's fast inference of Llama 3 (70B). This tool enables users to:
- Perform web searches with custom parameters
- Include their own source URLs
- Get AI-synthesized research summaries
- Ask follow-up questions about the sources
- Discover new relevant sources during Q&A

## Features
- Hybrid search combining user sources and web results
- Intelligent source management (user-provided vs. searched)
- Interactive Q&A with context awareness
- Citation tracking and source attribution
- Configurable search parameters
- Support for domain filtering

## Prerequisites

### API Keys Required
You'll need to obtain free API keys from:
1. **Groq** - For LLM inference
   - Sign up at: https://console.groq.com
   - Get API key from your dashboard
2. **Tavily** - For web search functionality
   - Sign up at: https://tavily.com
   - Get API key from your dashboard

### Environment Variables
Create a `.env` file in your project root with:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Installation

1. Clone the repository
2. Install requirements:
```
pip install -r requirements_for_groq_rag.txt
```

## Usage

Run the Streamlit app:
```
streamlit run tools/groq_web_search_RAG.py
```

The interface allows you to:
1. Enter a research query
2. Configure search parameters
3. Add your own source URLs
4. Get AI-synthesized research
5. Ask follow-up questions
6. See cited sources with full URLs

## Configuration Options
- Search depth (basic/advanced)
- Time range filter
- Content type filter
- Domain inclusion/exclusion
- Number of results per search
