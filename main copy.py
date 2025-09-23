from gpt_researcher.agent import GPTResearcher
import json
import os

from gpt_researcher.memory.embeddings import Memory

config_path = "./config/personal.json"
model = "hiveGPT_router:openai/gpt-oss-20b"
query = "Where is Milan?"


def create_llm_config_file(
    file_path: str,
    fast_llm: str,
    smart_llm: str,
    strategic_llm: str = None,
    embedding: str = "BAAI/bge-m3",
    retriever: str = "tavily",
    fast_token_limit: int = 3000,
    smart_token_limit: int = 6000,
    strategic_token_limit: int = 4000,
    temperature: float = 0.4,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    browse_chunk_max_length: int = 8192,
    summary_token_limit: int = 700,
):
    """
    Create a local GPT Researcher config JSON file with custom LLM and embedding.

    Args:
        file_path (str): Path to save the JSON file.
        fast_llm (str): Provider:model string for fast LLM.
        smart_llm (str): Provider:model string for smart LLM.
        strategic_llm (str, optional): Provider:model for strategic LLM.
        embedding (str): Embedding provider (default: OpenAI embedding).
        retriever (str): Retriever provider (default: "tavily").
        fast_token_limit, smart_token_limit, strategic_token_limit: Token limits.
        temperature (float): Temperature for LLMs.
        user_agent (str): HTTP User-Agent string.
        browse_chunk_max_length (int): Max chunk size for browsing.
        summary_token_limit (int): Max tokens for summary.
    """

    if not strategic_llm:
        strategic_llm = fast_llm  # default to fast if not provided

    config = {
        "RETRIEVER": retriever,
        "EMBEDDING": embedding,
        "SIMILARITY_THRESHOLD": 0.42,
        "FAST_LLM": fast_llm,
        "SMART_LLM": smart_llm,
        "STRATEGIC_LLM": strategic_llm,
        "FAST_TOKEN_LIMIT": fast_token_limit,
        "SMART_TOKEN_LIMIT": smart_token_limit,
        "STRATEGIC_TOKEN_LIMIT": strategic_token_limit,
        "BROWSE_CHUNK_MAX_LENGTH": browse_chunk_max_length,
        "CURATE_SOURCES": False,
        "SUMMARY_TOKEN_LIMIT": summary_token_limit,
        "TEMPERATURE": temperature,
        "USER_AGENT": user_agent,
        "MAX_SEARCH_RESULTS_PER_QUERY": 5,
        "MEMORY_BACKEND": "local",
        "TOTAL_WORDS": 1200,
        "REPORT_FORMAT": "APA",
        "MAX_ITERATIONS": 3,
        "AGENT_ROLE": None,
        "SCRAPER": "bs",
        "MAX_SCRAPER_WORKERS": 15,
        "MAX_SUBTOPICS": 3,
        "LANGUAGE": "english",
        "REPORT_SOURCE": "web",
        "DOC_PATH": "./my-docs",
        "PROMPT_FAMILY": "default",
        "LLM_KWARGS": {},
        "EMBEDDING_KWARGS": {},
        "VERBOSE": False,
        "DEEP_RESEARCH_BREADTH": 3,
        "DEEP_RESEARCH_DEPTH": 2,
        "DEEP_RESEARCH_CONCURRENCY": 4,
        "MCP_SERVERS": [],
        "MCP_AUTO_TOOL_SELECTION": True,
        "MCP_ALLOWED_ROOT_PATHS": [],
        "MCP_STRATEGY": "fast",
        "REASONING_EFFORT": "medium",
    }

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"✅ GPT Researcher config saved to {file_path}")

async def main():
    if model:
        create_llm_config_file(
            file_path=config_path,
            fast_llm=model,
            smart_llm=model,
            strategic_llm=model,
            embedding="hiveGPT_router:BAAI/bge-m3"
        )

    # Initialize GPT Researcher with tone parameter # Alternatively use "detailed_report"
    researcher = GPTResearcher(
        query,
        report_type="research_report",
        tone="objective",
        config_path=config_path  
    )

    await researcher.conduct_research()

    # Get the research context and sources
    context = researcher.get_research_context()
    sources = researcher.get_research_sources()
    source_urls = researcher.get_source_urls()
    
    print("\n--- Research Context ---")
    print(context)
    print("\n--- Research Sources ---")
    for src in sources:
        print(src)          
    print("\n--- Source URLs ---")
    for url in source_urls:
        print(url)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())