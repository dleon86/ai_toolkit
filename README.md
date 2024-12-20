# Comprehensive AI Toolkit

Welcome to the Comprehensive AI Toolkit project! This repository contains a suite of AI-powered tools designed to demonstrate proficiency with major AI platforms and frameworks. The toolkit includes tools for image processing, music generation, text-to-speech (TTS), speech-to-text (STT), various natural language processing (NLP) tasks, and more.

## **Features**

### **Image Processing**
- **SAM2 Image Segmentation:** Interactive tool for segmenting images using Meta's Segment Anything 2 (SAM2) model
  - Click to add foreground (left click) and background (right click) points
  - Save segmented regions with transparency
  - Clear selections and start over
- **More image tools coming soon...**

### **Groq Llama 3 Hybrid Search & Q&A**
A cutting-edge research assistant that combines user-provided sources with dynamic web search results, powered by Groq's Llama 3.3 and the pydantic_ai framework.
- **Hybrid Search:** Integrates user-supplied URLs with web search results for comprehensive research insights.
- **AI-Synthesized Summaries:** Leverages **Llama 3.3** by Groq to generate detailed and accurate research summaries.
- **Interactive Q&A:** Facilitates follow-up questions with context-aware answers using the **pydantic_ai** framework.
- **Advanced Search Configuration:** Customize search parameters including depth, time range, and domain filtering with **Tavily** integration.
- **Intelligent Source Management:** Differentiates between user-provided and searched sources for precise citations and references.

### **Coming Soon**
- **Music Generation:** Create original music compositions across different genres
- **Text-to-Speech & Speech-to-Text:** Convert text to speech and transcribe spoken words into text
- **NLP Tasks:** Perform text generation, summarization, and sentiment analysis

## **Getting Started**

### **Prerequisites**

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Git Bash (recommended for Windows users)
- Required Python packages listed in `requirements.txt` and `tools/requirements_groq_llama_search.txt`

#### **API Keys Required for Groq Llama 3 Hybrid Search & Q&A**
You'll need to obtain free API keys from:
1. **Groq** - For LLM inference
   - Sign up at: [Groq Console](https://console.groq.com)
   - Get API key from your dashboard
2. **Tavily** - For web search functionality
   - Sign up at: [Tavily](https://tavily.com)
   - Get API key from your dashboard

#### **Environment Variables**
Create a `.env` file in your project root with:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### **Installation**

1. **Clone the repository with submodules:**
    ```bash
    git clone --recursive https://github.com/dleon86/ai-toolkit.git
    cd ai-toolkit
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv ai_env
    source ai_env/bin/activate  # On Windows using Git Bash: source ai_env/Scripts/activate
    ```

3. **Install PyTorch with CUDA support (if available):**
    ```bash
    # For CUDA 11.8
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    # OR for CUDA 12.1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```

4. **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r tools/requirements_groq_llama_search.txt
    ```

5. **Install and setup SAM2:**
    ```bash
    # Initialize and update SAM2 submodule
    git submodule update --init --recursive
    
    # Install SAM2 requirements
    cd external/sam2
    pip install -e .
    
    # Download SAM2 model checkpoints
    ./download_ckpts.sh  # Use Git Bash on Windows
    cd ../..
    ```

6. **Install and setup Groq Llama 3 Hybrid Search & Q&A:**
    ```bash
    # Ensure environment variables are set in .env
    # No additional setup required beyond installing requirements
    ```

### **Usage**

#### **SAM2 Image Segmentation**
1. Navigate to the project root:
    ```bash
    cd path/to/ai-toolkit
    ```

2. Run the segmentation tool:
    ```bash
    python tools/image_processing_SAM2.py
    ```

3. Use the interactive interface:
   - Left click to mark foreground points
   - Right click to mark background points
   - Click "Clear" to reset selections
   - Click "Save" to export segmented regions with transparency

4. Find saved segments in the `saved_segments` directory:
   - `original_[timestamp].png`: Original image
   - `masked_[timestamp]_segment[n].png`: Selected regions with transparency
   - `unmasked_[timestamp]_segment[n].png`: Unselected regions with transparency

#### **Groq Llama 3 Hybrid Search & Q&A**
1. Navigate to the project root:
    ```bash
    cd path/to/ai-toolkit
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run tools/groq_web_search_RAG.py
    ```

3. Use the interactive interface to:
   - Enter a research query
   - Configure search parameters (depth, time range, content type, domains)
   - Add your own source URLs
   - Get AI-synthesized research summaries
   - Ask follow-up questions and view cited sources with full URLs

### **Contributing**

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

### **License**

This project is licensed under the MIT License.

---

**Stay tuned for more updates and tutorials!**
