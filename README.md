# MultiGraph-RAG System

[English](README.md) | [中文](README_zh.md)

---

## Overview

MultiGraph-RAG is an advanced Retrieval-Augmented Generation (RAG) system that combines knowledge graph extraction, vector search, and intelligent query routing to provide accurate and context-aware responses.

## Key Features

- **Multi-Source Data Loading**: Support for PDF, text files, and web content
- **Knowledge Graph Construction**: Automatic entity and relationship extraction using LLMs
- **Hybrid Retrieval**: Intelligent routing between vector search and graph traversal
- **Multi-Model Support**: Compatible with OpenAI, DeepSeek, DashScope, Ollama, and Claude
- **Flexible Embedding**: Support for HuggingFace, OpenAI, and local embeddings
- **Dual Database Architecture**: ChromaDB for vectors + Neo4j for knowledge graphs
- **Smart Caching**: Incremental indexing to avoid redundant processing
- **Interactive Chat**: Real-time Q&A with context-aware generation

## Architecture

```
Data Sources → Chunking → Graph Extraction → Dual Storage
                                              ↓
User Query → Router → Vector/Graph Retrieval → LLM Generation
```

**Components:**
- **Data Loaders**: PDF, text, web scraping
- **Semantic Splitter**: Markdown-aware chunking
- **Graph Builder**: Entity/relation extraction with embeddings
- **Query Router**: Local LLM-based retrieval strategy selection
- **Retrievers**: Vector similarity + Neo4j graph traversal
- **Generator**: Context-augmented response generation

## Prerequisites

- Python 3.13+
- Neo4j Database (local or cloud)
- GPU recommended for local models (optional)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MyRAG.git
cd MyRAG
```

2. **Install dependencies**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

3. **Set up Neo4j**
   - Download from [neo4j.com](https://neo4j.com/download/)
   - Start Neo4j and note your credentials
   - Default: `bolt://localhost:7687`

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

## Configuration

Edit `.env` file:

```bash
# Choose your LLM provider (openai/deepseek/dashscope/ollama/claude)
# Set corresponding API keys

# Neo4j credentials
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Model selection in src/graphrag/config.py
model_type = "claude"  # or "openai", "deepseek", etc.
embed_type = "huggingface"  # or "openai", "ollama"
```

## Usage

1. **Prepare your data**
```bash
mkdir -p data/input
# Place your PDF/text files in data/input/
```

2. **Run the system**
```bash
python main.py
```

3. **Interactive mode**
```
User Query: What is the main topic of the documents?
Model Answer: [Context-aware response based on your data]

User Query: exit
```

## Project Structure

```
MyRAG/
├── src/graphrag/
│   ├── config.py              # Configuration management
│   ├── data/                  # Data loading modules
│   │   ├── loader.py
│   │   ├── directory.py
│   │   └── loaders/           # PDF, text, web loaders
│   ├── db/                    # Database managers
│   │   ├── neo4j_manager.py
│   │   └── vector_store.py
│   ├── graph/
│   │   └── builder.py         # Knowledge graph extraction
│   ├── retrieval/             # Retrieval strategies
│   │   ├── router.py          # Query routing
│   │   ├── vector_retriever.py
│   │   └── graph_retriever.py
│   ├── generation/
│   │   └── generator.py       # Response generation
│   ├── splitters/
│   │   └── markdown.py        # Semantic chunking
│   └── utils/
│       └── state.py           # Index caching
├── main.py                    # Entry point
├── pyproject.toml             # Dependencies
└── .env.example               # Environment template
```

## How It Works

1. **Data Ingestion**: Documents are loaded and converted to markdown
2. **Semantic Chunking**: Text is split into meaningful segments
3. **Graph Extraction**: LLM extracts entities and relationships
4. **Dual Indexing**:
   - Text chunks → ChromaDB (vector search)
   - Entities/relations → Neo4j (graph traversal)
5. **Query Processing**:
   - Router analyzes query type
   - Selects vector/graph/hybrid retrieval
   - Retrieves relevant context
6. **Generation**: LLM generates answer using retrieved context

## Supported Models

**LLM Providers:**
- OpenAI (GPT-4, GPT-3.5)
- DeepSeek
- Alibaba DashScope (Qwen)
- Ollama (local models)
- Anthropic Claude

**Embedding Models:**
- HuggingFace Transformers (local)
- OpenAI Embeddings
- Ollama Embeddings

## Advanced Features

- **Multimodal Support**: Process images with vision models
- **Incremental Indexing**: Smart caching avoids reprocessing unchanged data
- **Flexible Routing**: Local small model for fast query classification
- **Tool Integration**: Weather API and web fetching capabilities

## Troubleshooting

**Neo4j Connection Error:**
- Ensure Neo4j is running: `neo4j status`
- Check credentials in `.env`

**API Key Issues:**
- Verify API keys are correctly set in `.env`
- Check API provider status

**Memory Issues:**
- Reduce `chunk_size` in `main.py`
- Use smaller embedding models

## Contributing

Contributions welcome! Please open issues or submit pull requests.

## License

MIT License
