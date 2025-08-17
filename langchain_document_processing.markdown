# LangChain Document Processing and RAG Pipeline Guide

This guide covers key LangChain components for loading, splitting, embedding, storing, and retrieving documents, along with advanced techniques like LCEL, LangServe, LangGraph, and agentic RAG workflows.

## 1. Document Loaders

### Text Loader
For loading plain `.txt` files from local storage.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/sample.txt")
documents = loader.load()
print(documents[0].page_content)  # View file content
```

### Web-Based Loader
For scraping and loading content from web pages.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
documents = loader.load()
for doc in documents:
    print(doc.metadata, doc.page_content[:200])
```

### PDF Loader
For extracting text from PDF files.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()
print(len(documents))  # Number of pages as separate docs
```

### S3 Directory Loader
For loading files from an AWS S3 bucket.

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

### Azure Blob Storage Loader
For loading files from an Azure Blob Storage container.

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
- Split text into chunks using `RecursiveCharacterTextSplitter`.
- Normalize metadata (e.g., source, page number).
- Use async loaders (`.aload()`) for large-scale ingestion.

## 2. Text Splitters

### RecursiveCharacterTextSplitter
Splits text into chunks without breaking words or sentences unnaturally.

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

**When to use**: General text (PDF, TXT, scraped content). Preserves semantic meaning.

### CharacterTextSplitter
Splits text purely by character count, ignoring sentence boundaries.

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

**When to use**: Structured text where splitting location doesn’t matter. Not ideal for semantic search.

### HTMLHeaderTextSplitter
Splits HTML based on header tags (`<h1>`, `<h2>`, etc.).

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """
<h1>Introduction</h1>
<p>This is intro text.</p>
<h2>Background</h2>
<p>Details about background.</p>
"""
headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html_text)
print(docs)
```

**When to use**: Web pages, blogs, or technical docs with HTML headings.

### RecursiveJsonSplitter
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

**When to use**: Large JSON logs, configs, or API responses.

**Summary Table**:

| Splitter Type               | Best For                     | Preserves Meaning? | Structure Aware? |
|-----------------------------|------------------------------|--------------------|------------------|
| RecursiveCharacterTextSplitter | General text                | ✅                 | ❌               |
| CharacterTextSplitter        | Raw text chunks             | ❌                 | ❌               |
| HTMLHeaderTextSplitter       | HTML docs                   | ✅                 | ✅               |
| RecursiveJsonSplitter        | JSON data                   | ✅                 | ✅               |

## 3. Embeddings

### OpenAI Embeddings
Hosted API, high-quality for semantic search, clustering, and RAG.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))  # embedding length
```

**Pros**: High accuracy, well-maintained.  
**Cons**: Requires API key, internet, paid after free credits.

### Hugging Face Embeddings
Open-source, many models available (e.g., `sentence-transformers/all-MiniLM-L6-v2`).

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))
```

**Pros**: Free, runs locally, many pre-trained models.  
**Cons**: Larger models can be slow without GPU.

### LLaMA-based Embeddings
Use fine-tuned LLaMA-based models (e.g., BGE, Instructor) via Ollama.

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama2")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))
```

**Pros**: Runs offline, customizable with fine-tuning.  
**Cons**: Needs strong hardware for large models.

**Comparison Table**:

| Provider        | Example Model                  | Dim  | Offline? | Cost | Quality |
|-----------------|-------------------------------|------|----------|------|---------|
| OpenAI          | text-embedding-3-large        | 3072 | ❌       | Paid | ⭐⭐⭐⭐⭐ |
| Hugging Face    | all-MiniLM-L6-v2             | 384  | ✅       | Free | ⭐⭐⭐⭐  |
| LLaMA-based     | BGE-large                    | 1024 | ✅       | Free | ⭐⭐⭐⭐  |

## 4. Vector Databases and Retrieval

### Vector Databases
Store embeddings for fast similarity search.

- **Chroma**: Open-source, local-first, good for prototyping.
  ```python
  from langchain_chroma import Chroma
  db = Chroma(collection_name="docs")
  ```

- **FAISS**: Fast, in-memory or disk, scales to millions of vectors.
  ```python
  from langchain_community.vectorstores import FAISS
  db = FAISS.from_texts(["doc1", "doc2"], embedding=embeddings)
  ```

### Retrieval Techniques
- **Similarity Search**: Finds closest vectors (cosine, dot product, L2).
  ```python
  results = db.similarity_search("query", k=5)
  ```

- **Max Marginal Relevance (MMR)**: Balances relevance and diversity.
  ```python
  results = db.max_marginal_relevance_search("query", k=5)
  ```

- **Filter-based Search**: Adds metadata filtering.
  ```python
  results = db.similarity_search("query", filter={"type": "pdf"})
  ```

**Why Retrieval?**  
- Pre-processes queries, post-processes results (re-ranking, filtering).  
- Integrates into LLM chains for contextual answering.

**L2 Score in Similarity Search**  
- L2 = Euclidean distance: `√∑(xi - yi)²`.  
- Lower L2 = more similar.  
- Used in FAISS for speed and simplicity in Euclidean space.

**all_dangerous_deserialization**  
- Allows loading pickled objects (e.g., FAISS index configs).  
- **Warning**: Only enable for trusted sources due to security risks.

## 5. LangChain Components

### LangChain OpenAI
Integrates OpenAI models into LangChain pipelines.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
response = llm.invoke("Write a haiku about the ocean")
print(response.content)
```

### LangChain Tracking
Tracks runs via LangSmith for debugging and monitoring.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"
```

### LangChain Project
Groups related runs/experiments in LangSmith.

```python
os.environ["LANGCHAIN_PROJECT"] = "financial-chatbot"
```

### ChatPromptTemplate
Creates structured, reusable prompts.

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question:\n{question}"
)
message = prompt.format_messages(question="What is the capital of France?")
print(message)
```

### Output Parser
Converts raw LLM output into structured formats (e.g., JSON).

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

## 6. Document Processing Chains

### create_stuff_document_chain
Combines retrieved documents into a single prompt for LLM.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
Answer the question based on the following documents:
{context}

Question: {question}
""")
chain = create_stuff_documents_chain(llm, prompt)
docs = [
    {"page_content": "Paris is the capital of France."},
    {"page_content": "It is known for the Eiffel Tower."}
]
response = chain.invoke({"question": "What is the capital of France?", "context": docs})
print(response)
```

**Pros**: Simple, works for small contexts.  
**Cons**: Token overflow for large documents.

## 7. LangChain Expression Language (LCEL)

LCEL composes LangChain components into pipelines.

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
- Clean, functional syntax.  
- Supports `.invoke()`, `.batch()`, `.stream()`.

## 8. LangServe

Deploys LangChain pipelines as APIs via FastAPI.

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

## 9. LangGraph Concepts

### Reducer
Merges state updates from multiple nodes (e.g., concatenate lists).

```python
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], operator.add]
```

### Stream
Enables real-time, token-level outputs or intermediate steps.

```python
async for chunk in app.stream(input_data):
    print(chunk)
    app.update({"messages": chunk["messages"]})
```

### Dataclasses
Define graph state schema for type safety.

### Router
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

### Tool Binding
Connects external functions to agents.

```python
from langchain.tools import tool

@tool
def get_weather(city: str):
    return f"The weather in {city} is sunny."
agent = agent.bind_tools([get_weather])
```

### Agent Memory
Stores conversation history or intermediate results.

## 10. Advanced RAG Techniques

### Agentic RAG
Agent dynamically decides when/how to retrieve.

### Corrective RAG
Model self-reflects and grades its answers, re-retrieves if needed.

### Adaptive RAG
Adjusts retrieval strategy based on query complexity.

**Summary Table**:

| Type            | Core Idea                           | Purpose                      | Example                      |
|-----------------|-------------------------------------|------------------------------|------------------------------|
| Agentic RAG     | Agent decides retrieval strategy    | Smarter retrieval control    | Multi-step reasoning Q&A     |
| Corrective RAG  | Self-reflection & grading           | Reduce hallucination         | Validate generated answers   |
| Adaptive RAG    | Adjust retrieval dynamically        | Balance accuracy vs speed    | Adjust docs per query        |

This unified pipeline can load from TXT, PDF, HTML, JSON, S3, or Azure Blob, automatically select the appropriate splitter, generate embeddings, store in a vector database, and support advanced RAG workflows.