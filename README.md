# Agent Notebook (RAG + Web Search) — README

This notebook builds a small **agentic Q&A system** using LangChain:

- **RAG Agent**: answers questions using only retrieved passages from a vector store.
- **Web Search Agent**: answers questions using Tavily-powered web search.
- **Hybrid Agent**: selects the right tool (retrieval vs. web search) based on the question.

The demo corpus is a single web article loaded from:

- Lilian Weng, “Agents” (2023-06-23): https://lilianweng.github.io/posts/2023-06-23-agent/

---

## Key components

### Models
- **Chat model**: `ChatOpenAI(model="gpt-4o")`
- **Embeddings**: `OpenAIEmbeddings(model="text-embedding-3-large")`

### Vector store
- Default: `InMemoryVectorStore(embeddings)`
- Optional (commented in notebook): Chroma persistence via `langchain-chroma`

### Document ingestion
1. Load HTML from the blog post using `WebBaseLoader`.
2. Parse only relevant sections using `bs4.SoupStrainer`.
3. Chunk the text with `RecursiveCharacterTextSplitter`:
   - `chunk_size=1000`
   - `chunk_overlap=200`
   - `add_start_index=True`
4. Add chunks to the vector store.

### Tools
- **`retrieve_context`**: a LangChain `@tool` that runs `vector_store.similarity_search(query, k=2)` and returns:
  - a serialized text context (with metadata), and
  - the retrieved document objects as an artifact.
- **`TavilySearch`**: web search tool with `max_results=20`.

### Agents
The notebook uses `langchain.agents.create_agent(...)` to create:
- a retrieval-only agent (strict “use retrieved context only” instructions),
- a web-search agent, and
- a hybrid agent that can use both tools.

### Interactive chat
A `chat_with_agent(hybrid_agent)` function provides a simple terminal-style chat loop and streams responses.

---

## Prerequisites

- Python 3.9+ (or Google Colab)
- An **OpenAI API key**
- A **Tavily API key** (only required for the web-search and hybrid sections)

---

## Setup

### Option A: Run in Google Colab (as written)
The notebook uses:

```python
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = userdata.get("TAVILY_API_KEY")
```

Add the corresponding secrets in Colab (Runtime → Secrets) and run the cells top-to-bottom.

### Option B: Run locally (recommended)
1. Export environment variables:

```bash
export OPENAI_API_KEY="..."
export TAVILY_API_KEY="..."   # optional if you skip Tavily sections
```

2. Install dependencies (the notebook installs these inline):

```bash
pip install -U langchain langchain-openai langchain-core langchain-text-splitters langchain-community bs4 langchain-tavily
```

---

## How to use

1. **Index the document**: run the data loading, splitting, and vector-store ingestion cells.
2. **Create an agent**:
   - Retrieval-only: `tools = [retrieve_context]`
   - Web-only: `tools = [tavily_tool]`
   - Hybrid: `tools = [retrieve_context, tavily_tool]`
3. **Ask a question** using agent streaming:

```python
query = "What is the standard method for Task Decomposition?"
for event in hybrid_agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

4. **Chat interactively**:

```python
chat_with_agent(hybrid_agent)
```

Type `bye` to exit.

---

## Notes and tips

- The retrieval-only prompt is intentionally strict; if the retrieved chunks do not contain the answer, the agent should respond with:
  - `I have no context for this.`
- If you enable the Chroma section, you can persist the vector store to disk (`persist_directory`) instead of keeping it in memory.
- For better RAG quality on larger corpora, consider:
  - increasing `k` in similarity search,
  - using metadata filters,
  - adding reranking, or
  - normalizing/cleaning HTML more aggressively.

---

## Troubleshooting

- **Authentication errors**: confirm `OPENAI_API_KEY` (and `TAVILY_API_KEY` if using Tavily) are set in your environment.
- **Package version issues**: restart the kernel/runtime after installs, especially in Colab.
- **Empty/irrelevant retrieval**: confirm documents were split and added to the vector store before querying.

---

## License

This README describes a learning/demo notebook. Ensure you comply with the licenses/terms of any external content, APIs, and models you use.
