# LLM Application Components

## 1. Vector Stores

### What They Are
Vector stores are specialized databases designed to store dense vector representations (embeddings) of data such as text, images, or audio. These vectors enable semantic search using similarity measures like cosine similarity.

### Why They're Used
- To retrieve semantically relevant information based on user queries.
- Essential in Retrieval-Augmented Generation (RAG) workflows to provide relevant context to large language models (LLMs).

### Popular Vector Databases
| Vector Store | Features |
|--------------|----------|
| FAISS        | Fast, CPU/GPU, open-source, local |
| Pinecone     | Managed, scalable, persistent |
| Weaviate     | Graph+vector hybrid, RESTful API |
| Chroma       | Lightweight, easy to use (local/dev) |
| Milvus       | High-performance, cloud-ready |
| Qdrant       | Rust-based, real-time vector search |

## 2. Memory Modules

### What They Are
Memory modules enable LLM applications to retain past interactions, storing chat history or retrieved data either temporarily (in-session) or permanently (long-term memory).

### Why They're Used
- To create stateful conversations.
- To maintain context continuity across turns.
- To enable personalized experiences in chatbots and agents.

### Types of Memory in LLM Apps
| Type                | Description                                      |
|---------------------|--------------------------------------------------|
| Short-Term Memory   | In-session memory stored in RAM                  |
| Long-Term Memory    | Persistent memory via vector stores/databases    |
| ConversationBuffer  | Stores raw chat history                          |
| SummaryMemory       | Stores summarized version of past dialogue       |
| CombinedMemory      | Mixes context from multiple memory sources       |

*Libraries like LangChain or LlamaIndex provide memory abstractions.*

## 3. LLM App Frameworks

### What They Are
Frameworks that simplify building end-to-end applications with LLMs, integrating memory, tools, vector stores, APIs, and user interfaces.

### Popular Frameworks
| Framework         | Use Case                     | Features                                  |
|-------------------|------------------------------|-------------------------------------------|
| LangChain         | Chain LLMs with tools/memory | Agent, RAG, Vector search, Chains, Memory |
| LlamaIndex        | RAG-focused document interaction | Indexing, Query Engines, Agents        |
| Haystack          | NLP pipelines & QA systems   | Modular, ElasticSearch support           |
| Semantic Kernel   | Microsoft’s .NET-based agent framework | Planning, memory, skills support |
| Guidance          | Prompt templating            | Fine prompt control using templates      |

### What They Help You Do
- Create retrieval pipelines (RAG).
- Implement agents with tools.
- Connect LLMs to external APIs.
- Use embeddings + vector databases for context.
- Maintain chat memory.

## How They All Work Together
A typical LLM-based application (e.g., a smart chatbot) might:
1. Embed the user query.
2. Retrieve relevant chunks from a vector store.
3. Pass context and query to the LLM.
4. Use memory to retain past conversations.
5. Serve the application via frameworks like LangChain or LlamaIndex.