# Comprehensive LangChain Document Processing and RAG Pipeline Guide

This guide provides a complete overview of LangChain components for document loading, text splitting, embedding generation, vector storage, retrieval, and advanced techniques like LCEL, LangServe, LangGraph, and agentic RAG workflows. It includes all provided examples, explanations, and concepts for building a unified ingestion pipeline.

## 1. Document Loaders

Document loaders fetch and parse data from various sources into LangChain's document format.

### 1.1 Text Loader
Loads plain `.txt` files from local storage.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/sample.txt")
documents = loader.load()
print(documents[0].page_content)  # View file content
```

### 1.2 Web-Based Loader
Scrapes and loads content from web pages.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
documents = loader.load()
for doc in documents:
    print(doc.metadata, doc.page_content[:200])
```

### 1.3 PDF Loader
Extracts text from PDF files, treating each page as a separate document.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()
print(len(documents))  # Number of pages as separate docs
```

### 1.4 S3 Directory Loader
Loads files from an AWS S3 bucket.

```python
from langchain_community.document_loaders import S3DirectoryLoader

loader = S3DirectoryLoader(
    bucket="my-bucket-name",
    prefix="data/documents/",  # folder path inside S3
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET"
)
documents = loader.load()
```

### 1.5 Azure Blob Storage Loader
Loads files from an Azure Blob Storage container.

```python
from langchain_community.document_loaders import AzureBlobStorageFileLoader

loader = AzureBlobStorageFileLoader(
    conn_str="DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net",
    container="my-container",
    blob_name="sample.pdf"
)
documents = loader.load()
```

**Tips for All Loaders**:
- After loading, split text into chunks using `RecursiveCharacterTextSplitter`.
- Normalize metadata (e.g., source, page number) for consistency.
- Use async loaders (`.aload()`) for large-scale ingestion to improve performance.

**Unified Pipeline Option**: A single pipeline can load from TXT, Web, PDF, S3, or Azure Blob, automatically select the right splitter, and store into a vector database for retrieval-augmented generation (RAG).

## 2. Text Splitters

Text splitters break large documents into smaller chunks for embedding and retrieval.

### 2.1 RecursiveCharacterTextSplitter
Splits text without breaking words or sentences, using multiple separators in priority order (paragraph → sentence → word → character).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Max characters per chunk
    chunk_overlap=50,    # Overlap between chunks for context
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_text("Your large text here...")
print(docs)
```

**When to use**:
- ✅ General text (PDF, TXT, scraped content).
- ✅ Preserves semantic meaning by avoiding mid-word breaks.
- Best for most document types.

### 2.2 CharacterTextSplitter
Splits text purely by character count, ignoring sentence or paragraph boundaries.

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"  # split at newline if possible
)
docs = text_splitter.split_text("Your large text here...")
print(docs)
```

**When to use**:
- ✅ Structured or semi-structured text where splitting location doesn’t matter.
- ❌ Not ideal for conversational or semantic search due to potential mid-sentence breaks.

### 2.3 HTMLHeaderTextSplitter
Parses HTML and splits based on header tags (`<h1>`, `<h2>`, etc.), keeping sections under their respective headers.

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """
<h1>Introduction</h1>
<p>This is intro text.</p>
<h2>Background</h2>
<p>Details about background.</p>
"""
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2")
]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html_text)
print(docs)
```

**When to use**:
- ✅ Web pages, blog posts, technical documentation with HTML headings.
- ✅ Keeps context grouped by topic.

### 2.4 RecursiveJsonSplitter
Splits nested JSON into smaller pieces while preserving structure.

```python
from langchain_text_splitters import RecursiveJsonSplitter

json_data = {
    "user": {
        "name": "Alice",
        "bio": "Data scientist with experience in ML.",
        "projects": [
            {"name": "Project A", "desc": "AI model"},
            {"name": "Project B", "desc": "Data pipeline"}
        ]
    }
}
splitter = RecursiveJsonSplitter(max_chunk_size=50)
docs = splitter.split_json(json_data)
print(docs)
```

**When to use**:
- ✅ Large JSON logs, configs, or API responses.
- ✅ Keeps related key-value pairs together.

**Summary Table**:

| Splitter Type               | Best For                     | Preserves Meaning? | Structure Aware? |
|-----------------------------|------------------------------|--------------------|------------------|
| RecursiveCharacterTextSplitter | General text                | ✅                 | ❌               |
| CharacterTextSplitter        | Raw text chunks             | ❌                 | ❌               |
| HTMLHeaderTextSplitter       | HTML docs                   | ✅                 | ✅               |
| RecursiveJsonSplitter        | JSON data                   | ✅                 | ✅               |

**Unified Pipeline Note**: A single ingestion pipeline can automatically choose the right splitter based on file type (e.g., `HTMLHeaderTextSplitter` for HTML, `RecursiveJsonSplitter` for JSON).

## 3. Embeddings

Embeddings convert text into numerical vectors for similarity search and RAG.

### 3.1 OpenAI Embeddings
Hosted API, optimized for semantic search, clustering, and RAG. Common models: `text-embedding-3-small`, `text-embedding-3-large`.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))  # embedding length
```

**Pros**:
- ✅ High accuracy.
- ✅ Well-maintained.
**Cons**:
- ❌ Requires API key and internet.
- ❌ Paid after free credits.

### 3.2 Hugging Face Embeddings
Open-source, free, with many models (e.g., `sentence-transformers/all-MiniLM-L6-v2`).

**Local Example**:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))
```

**Hugging Face Inference API Example**:

```python
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key="YOUR_HF_API_KEY",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))
```

**Pros**:
- ✅ Free, runs locally.
- ✅ Many pre-trained models.
**Cons**:
- ❌ Larger models can be slow without GPU.

### 3.3 LLaMA-based Embeddings
Uses fine-tuned LLaMA-based models (e.g., BGE, Instructor) via Ollama.

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama2")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))
```

**Example Models**:
- `BAAI/bge-large-en-v1.5`: High quality.
- `hkunlp/instructor-xl`: Task-aware embeddings.

**Pros**:
- ✅ Runs fully offline.
- ✅ Customizable with fine-tuning.
**Cons**:
- ❌ Needs strong hardware for big models.

**Comparison Table**:

| Provider        | Example Model                  | Dim  | Offline? | Cost | Quality |
|-----------------|-------------------------------|------|----------|------|---------|
| OpenAI          | text-embedding-3-large        | 3072 | ❌       | Paid | ⭐⭐⭐⭐⭐ |
| Hugging Face    | all-MiniLM-L6-v2             | 384  | ✅       | Free | ⭐⭐⭐⭐  |
| LLaMA-based     | BGE-large                    | 1024 | ✅       | Free | ⭐⭐⭐⭐  |

## 4. Vector Databases and Retrieval

Vector databases store embeddings and enable fast similarity searches.

### 4.1 Vector Databases

- **Chroma**: Open-source, local-first, ideal for prototyping RAG pipelines.
  ```python
  from langchain_chroma import Chroma
  db = Chroma(collection_name="docs")
  ```

- **FAISS**: Fast, in-memory or disk-based, scales to millions of vectors.
  ```python
  from langchain_community.vectorstores import FAISS
  db = FAISS.from_texts(["doc1", "doc2"], embedding=embeddings)
  ```

- **Hugging Face Hub Datasets**: Stores precomputed embeddings, good for sharing public datasets (not live search).

### 4.2 Retrieval Techniques

- **Similarity Search**: Finds vectors closest to the query vector (cosine, dot product, L2).
  ```python
  results = db.similarity_search("query", k=5)
  ```

- **Max Marginal Relevance (MMR)**: Balances relevance and diversity to avoid redundant results.
  ```python
  results = db.max_marginal_relevance_search("query", k=5)
  ```

- **Filter-based Search**: Filters results by metadata (e.g., document type, date).
  ```python
  results = db.similarity_search("query", filter={"type": "pdf"})
  ```

**Why Retrieval?**  
- Pre-processes queries (rephrasing, embedding creation).  
- Post-processes results (re-ranking, filtering, metadata enrichment).  
- Integrates with LLM chains for contextual answering.

**L2 Score in Similarity Search**  
- L2 = Euclidean distance: `√∑(xi - yi)²`.  
- Lower L2 = more similar.  
- Used in FAISS for speed and simplicity in Euclidean space, especially when vectors are not normalized.

**all_dangerous_deserialization**  
- Allows loading arbitrary pickled Python objects (e.g., FAISS index configs).  
- **Warning**: Dangerous due to potential execution of arbitrary code. Only enable for trusted sources. Prefer JSON for metadata.

## 5. LangChain Components

### 5.1 LangChain OpenAI
Integration layer for OpenAI APIs (chat models, embeddings). Handles API authentication, prompt formatting, token counting, and retry logic.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
response = llm.invoke("Write a haiku about the ocean")
print(response.content)
```

### 5.2 LangChain Tracking
Tracks prompts, inputs, outputs, errors, and timing via LangSmith for debugging and monitoring.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"
```

### 5.3 LangChain Project
Groups related runs, datasets, and experiments in LangSmith for a specific application.

```python
os.environ["LANGCHAIN_PROJECT"] = "financial-chatbot"
```

### 5.4 ChatPromptTemplate
Creates structured, reusable prompts with placeholders for dynamic values.

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question:\n{question}"
)
message = prompt.format_messages(question="What is the capital of France?")
print(message)
```

### 5.5 Output Parser
Converts raw LLM output into structured formats (e.g., JSON, lists, Python objects).

```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

schemas = [
    ResponseSchema(name="answer", description="The answer to the question"),
    ResponseSchema(name="source", description="Where the answer comes from")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
prompt_text = parser.get_format_instructions()
print(prompt_text)
```

**Summary Table**:

| Term                  | Purpose                                              |
|-----------------------|-----------------------------------------------------|
| LangChain OpenAI      | Integrates OpenAI models into LangChain pipelines    |
| LangChain Tracking    | Logs & monitors chain executions (via LangSmith)    |
| LangChain Project     | Groups related runs/experiments into one workspace   |
| ChatPromptTemplate    | Creates dynamic, reusable prompts for chat models    |
| Output Parser         | Converts messy LLM output into structured data       |

## 6. Document Processing Chains

### 6.1 create_stuff_document_chain
Combines multiple retrieved documents into a single prompt for the LLM.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Step 1: Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Step 2: Define Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
Answer the question based on the following documents:
{context}

Question: {question}
""")

# Step 3: Create Stuff Chain
chain = create_stuff_documents_chain(llm, prompt)

# Step 4: Run the chain
docs = [
    {"page_content": "Paris is the capital of France."},
    {"page_content": "It is known for the Eiffel Tower."}
]
response = chain.invoke({"question": "What is the capital of France?", "context": docs})
print(response)
```

**Why “Stuff” Approach?**  
- **Pros**: Simple, direct—combines all context into one prompt.  
- **Cons**: Can fail with large documents due to token limits; no ranking or summarization.  

**Other Strategies**:
- **Map-Reduce**: Summarize each document, then summarize summaries.
- **Refine**: Generate an answer from the first document, iteratively refine with others.

## 7. LangChain Expression Language (LCEL)

LCEL is a declarative way to compose LangChain components (LLMs, prompts, retrievers, parsers) into pipelines.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
parser = StrOutputParser()
chain = prompt | llm | parser
result = chain.invoke({"text": "Hello World"})
print(result)  # Bonjour le monde
```

**Why LCEL?**  
- Clean, functional-style syntax.  
- Easier debugging and testing.  
- Supports `.invoke()`, `.batch()`, `.stream()` for flexible execution.

**LangChain Core**  
- Minimal, dependency-light foundation of LangChain.  
- Includes base classes for LLMs, prompts, document loaders, output parsers, and tracing.  
- **Why use it?**: Lightweight, faster imports, easier to build custom components.

**Why OutputParser?**  
- Converts messy LLM output into structured formats (e.g., JSON, Pydantic models).  
- Prevents formatting hallucinations and ensures machine-readable outputs.

```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

schemas = [
    ResponseSchema(name="answer", description="The main answer"),
    ResponseSchema(name="source", description="Where the answer came from")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()
prompt_text = f"""
Answer the question and respond in the format below:
{format_instructions}

Question: What is the capital of France?
"""
```

## 8. LangServe

LangServe deploys LangChain pipelines as APIs using FastAPI.

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

app = FastAPI()
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm
add_routes(app, chain, path="/joke")
```

**Usage**: POST to `/joke` with `{"topic": "cats"}`.  
**Benefits**:  
- Auto-generated API docs (Swagger/OpenAPI).  
- Supports streaming, batching, and authentication.  
- Makes chains callable by external apps (web, mobile, CLI).

**Chains as API**: Exposes LangChain chains as endpoints for integration with frontends (React, Streamlit) or other systems.

**trim_messages**  
- Shortens conversation history to fit LLM token limits.  
- Can remove oldest messages, truncate long messages, or preserve system prompts.

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import trim_messages

messages = [
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!"),
    HumanMessage(content="Tell me a long story about AI...")
]
trimmed = trim_messages(messages, max_tokens=100, strategy="last")
```

## 9. LangGraph Concepts

LangGraph is a framework for building dynamic, stateful workflows with nodes and edges.

### 9.1 Reducer
Merges state updates from multiple nodes using custom logic (e.g., list concatenation).

```python
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], operator.add]
```

**Example**: If one node updates `bar` with `["A"]` and another with `["B"]`, the reducer combines them into `["A", "B"]`.

### 9.2 Annotation Function
Attaches a reducer to a state key using `Annotated`.

```python
messages: Annotated[list[AnyMessage], add_messages]
```

**Purpose**: `add_messages` reducer appends or updates messages intelligently, handling IDs and deserialization.

### 9.3 Stream
Enables real-time, token-level outputs or intermediate step visualization.

```python
async for chunk in app.stream(input_data):
    print(chunk)
    app.update({"messages": chunk["messages"]})
```

**Benefits**: Real-time reasoning, progress tracking, human intervention.

### 9.4 Dataclasses
Define graph state schema for type safety (alternative to `TypedDict` or Pydantic models).

### 9.5 ChatMsg (Messages)
Stores conversation history as `HumanMessage`, `AIMessage`, or `AnyMessage`, managed by `add_messages` reducer.

### 9.6 LLM in Graph Nodes
Integrates LLMs into nodes to process state and produce outputs.

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

def ask_model(state: State):
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke({"inputs": state["messages"]})
    return {"messages": [AIMessage(content=response.content)]}
```

### 9.7 Tool Binding
Binds external functions as tools for LLMs to call.

```python
from langchain.tools import tool

@tool
def get_weather(city: str):
    return f"The weather in {city} is sunny."
agent = agent.bind_tools([get_weather])
```

### 9.8 Tool Calls from LLM
LLM decides to call tools, and nodes handle execution and state updates.

**Example Workflow**:
- LLM outputs intent (e.g., call `get_weather` with `city="Delhi"`).
- Node invokes tool and updates state with results.

### 9.9 Router
Directs flow to nodes based on conditions.

```python
from langgraph.graph import StateGraph

graph = StateGraph()
def route(state):
    if "math" in state["query"]:
        return "math_node"
    else:
        return "chat_node"
graph.add_router("router_node", route)
```

### 9.10 Conditional Edges
Routes from one node to another based on conditions.

```python
graph.add_conditional_edges(
    "router_node",
    condition=lambda state: state["intent"] == "math",
    target="math_node"
)
```

### 9.11 Agent Architecture
- **ReAct**: Reason → Act → Observe → Repeat.
  - Example: Reason about query, call tool (e.g., `get_weather`), observe result, answer.
- **Plan-and-Execute**: LLM plans steps, then executes.
- **Self-Ask with Search**: Agent clarifies questions before acting.
- **Tool-Calling Agents**: LLM outputs structured JSON for tool calls.

### 9.12 Agent Memory
Stores context or intermediate results.
- **ConversationBufferMemory**: Raw chat history.
- **ConversationSummaryMemory**: Summarized history to save tokens.
- **VectorStoreRetrieverMemory**: Stores/retrieves facts via embeddings.

**Example**: Remembers “flight to New York” when user later says “Make it business class.”

### 9.13 astream
Asynchronous streaming for non-blocking, real-time outputs.

```python
async for event in agent.astream({"query": "What's the weather?"}):
    print(event)
```

## 10. Advanced Workflow Techniques

### 10.1 Prompt Chaining
Breaks complex tasks into sequential prompts (Generate → Improve → Polish).

**Example Flow**:
- Generate: Raw draft (e.g., blog).
- Improve: Refine style, grammar, facts.
- Polish: Ensure tone, conciseness, formatting.

**Purpose**: Improves quality, reduces hallucinations.

### 10.2 Parallelization
Runs multiple chains/nodes concurrently for speed.

**Example**: Embed documents, call APIs, or run graph branches in parallel.

### 10.3 Routing in LangGraph
Directs flow dynamically based on conditions (e.g., math query → calculator node).

### 10.4 Orchestrator Workflow
Manages entire conversation flow:
- Decides which nodes/tools run.
- Handles branching and parallel execution.
- Collects results.

### 10.5 Send API
Passes messages/data between nodes in dynamic, asynchronous workflows.

```python
await graph.send("node_name", input_data)
```

### 10.6 Evaluator / Optimizer
- **Evaluator**: Measures performance (e.g., accuracy, BLEU score).
- **Optimizer**: Adjusts prompts, retrieval, or LLM settings based on feedback.

**Example**: Evaluate answer correctness, adjust retrieval strategy if score is low.

## 11. Advanced RAG Techniques

### 11.1 Agentic RAG
Combines RAG with agent reasoning:
- Dynamically decides when/what to retrieve.
- Supports multi-step retrieval and reasoning.

**Use Case**: Customer support bots reasoning over multiple documents.

### 11.2 Corrective RAG
Model self-reflects and grades answers:
- Critiques for factual correctness or missing context.
- Re-retrieves or regenerates if score is low.

**Example**: Secondary prompt grades answer, triggers re-retrieval if needed.

### 11.3 Adaptive RAG
Adjusts retrieval strategy based on query complexity:
- Simple queries: Fewer documents, faster response.
- Complex queries: More retrieval steps, multi-hop retrieval.

**Methods**:
- Adaptive `k`-value: Retrieve more docs for vague queries.
- Adaptive reranking: Use stronger reranker for complex cases.
- Query reformulation: Rewrite query if retrieval fails.

**Summary Table**:

| Type            | Core Idea                           | Purpose                      | Example                      |
|-----------------|-------------------------------------|------------------------------|------------------------------|
| Agentic RAG     | Agent decides retrieval strategy    | Smarter retrieval control    | Multi-step reasoning Q&A     |
| Corrective RAG  | Self-reflection & grading           | Reduce hallucination         | Validate generated answers   |
| Adaptive RAG    | Adjust retrieval dynamically        | Balance accuracy vs speed    | Adjust docs per query        |

## 12. Unified Ingestion Pipeline

A production-ready pipeline can:
- Load from TXT, PDF, HTML, JSON, S3, Azure Blob.
- Automatically select the appropriate splitter (`RecursiveCharacterTextSplitter` for text/PDF, `HTMLHeaderTextSplitter` for HTML, etc.).
- Generate embeddings using OpenAI, Hugging Face, or LLaMA-based models.
- Store embeddings in a vector database (Chroma, FAISS).
- Support retrieval with similarity search, MMR, or filter-based methods.
- Integrate with LCEL for chaining, LangServe for API deployment, and LangGraph for dynamic workflows.
- Enable advanced RAG (agentic, corrective, adaptive) for robust question-answering.

This pipeline ensures flexibility, scalability, and production-readiness for RAG applications.
