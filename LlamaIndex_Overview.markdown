# LlamaIndex Overview

## 1. What is LlamaIndex?

LlamaIndex is a framework that connects Large Language Models (LLMs) to your private data.

- LLMs (like GPT, LLaMA, Mistral) are trained on general data.
- But in production, you want them to answer using your own documents, PDFs, databases, APIs, etc.
- LlamaIndex builds a data pipeline + index, so the LLM can retrieve the right context at query time.

## 2. LlamaIndex Architecture

### Layers:
- **Data Ingestion (Loaders & Parsers)**
  - Input sources: PDFs, Word docs, Notion, Google Drive, APIs, SQL, etc.
  - Tools: `SimpleDirectoryReader`, `LlamaParse` (structured parsing).
- **Data Indexing**
  - Converts docs into nodes (chunks).
  - Builds indexes: Vector (embeddings), Keyword, Graph, List.
- **Retrieval & Query**
  - Finds the most relevant chunks at query time.
  - Query engine = Retrieval + LLM synthesis (RAG pipeline).
- **Application Layer (Chat/Agents)**
  - Interfaces for users.
  - Chatbots, QA systems, multi-doc exploration, agents that call tools.

### Conceptual Flow (Text Diagram)
```
[User Query]
    ↓
[Application Layer: Chat / Agent]
    ↓
[Query Engine: Retrieval + LLM synthesis]
    ↓
[Index: Vector / Graph / List]
    ↓
[Data Ingestion: Loaders / Parsers]
    ↓
[Your Data Sources: PDFs, DBs, APIs]
```

**So, LlamaIndex = end-to-end pipeline from your data → index → retrieval → LLM answers.**

## 3. Persist in LlamaIndex

Normally, indexes exist only in memory. `persist` allows you to save embeddings + nodes + index to disk and reload later.

### Why important?
- In production, you don’t want to recompute embeddings every time.

### Example Code
```python
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader

# Step 1: Load data
docs = SimpleDirectoryReader("data/").load_data()

# Step 2: Build index
index = VectorStoreIndex.from_documents(docs)

# Step 3: Persist
index.storage_context.persist(persist_dir="./storage")

# Step 4: Reload later
storage_context = StorageContext.from_defaults(persist_dir="./storage")
new_index = load_index_from_storage(storage_context)
```

## 4. LlamaParse (vs. SimpleDirectoryReader)

| Feature                  | SimpleDirectoryReader         | LlamaParse (advanced)         |
|--------------------------|-------------------------------|-------------------------------|
| **File types**           | TXT, PDF, DOCX, MD           | PDF (complex)                |
| **Parsing quality**      | Basic text                   | Structured, accurate          |
| **Tables/images handling** | ❌ No                      | ✅ Yes                      |
| **Metadata extraction**  | Limited                      | Rich (titles, tables)        |
| **Free/Paid**            | Free (local)                 | Paid (API-based)             |

### Example – SimpleDirectoryReader
```python
from llama_index.core import SimpleDirectoryReader
docs = SimpleDirectoryReader("./data").load_data()
```

### Example – LlamaParse
```python
from llama_parse import LlamaParse

parser = LlamaParse(api_key="YOUR_API_KEY", result_type="markdown")
docs = parser.load_data("contracts/contract1.pdf")
```

**Think**: `SimpleDirectoryReader` = quick/basic, `LlamaParse` = enterprise-grade parsing.

## 5. Integrating Cloud Storage

### Load Documents from S3

#### Option A: S3Reader
```python
from llama_index.readers.s3 import S3Reader

s3_reader = S3Reader(
    bucket="my-bucket-name",
    prefix="docs/",
    aws_region="us-east-1"
)
docs = s3_reader.load_data()
```

#### Option B: boto3 + SimpleDirectoryReader
```python
import boto3, os
from llama_index.core import SimpleDirectoryReader

s3 = boto3.client('s3')
bucket = "my-bucket-name"
key = "docs/myfile.pdf"
local_path = "downloads/myfile.pdf"

os.makedirs("downloads", exist_ok=True)
s3.download_file(bucket, key, local_path)

docs = SimpleDirectoryReader("downloads").load_data()
```

### Load Documents from Azure Blob
```python
from azure.storage.blob import BlobServiceClient
from llama_index.core import SimpleDirectoryReader
import os

connect_str = "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=..."
container_name = "mycontainer"
blob_name = "docs/myfile.pdf"

blob_service = BlobServiceClient.from_connection_string(connect_str)
blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)

os.makedirs("downloads", exist_ok=True)
local_path = os.path.join("downloads", "myfile.pdf")

with open(local_path, "wb") as f:
    f.write(blob_client.download_blob().readall())

docs = SimpleDirectoryReader("downloads").load_data()
```

## 6. Vector Database (ChromaDB)

### Store Embeddings in ChromaDB
```bash
pip install chromadb llama-index-vector-stores-chroma
```

```python
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Step 1: Load documents
docs = SimpleDirectoryReader("downloads").load_data()

# Step 2: Setup ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")  
chroma_collection = chroma_client.get_or_create_collection("my_docs")

# Step 3: Create vector store + storage context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Step 4: Build index (embeddings saved in ChromaDB)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# Step 5: Query
query_engine = index.as_query_engine()
response = query_engine.query("What does the document say about pricing?")
print(response)
```

### Reload Index from ChromaDB
```python
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store)
```

## 7. Final Summary

- **LlamaIndex** = framework for connecting LLMs to your private data (PDFs, APIs, DBs).
- **Architecture** = Ingestion → Index → Retrieval → Application.
- **Persist** = save/reload embeddings + indexes (production ready).
- **LlamaParse** = advanced PDF parser (upgrade over `SimpleDirectoryReader`).
- **Cloud Storage**:
  - **AWS** → `S3Reader` or `boto3`.
  - **Azure** → `azure-storage-blob`.
- **ChromaDB** = persistent vector database for embeddings (integrates natively).