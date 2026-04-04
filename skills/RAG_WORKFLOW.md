# RAG Literature Search Workflow

## Basic Information
- Endpoints: /api/v1/rag/search (GET), /api/v1/rag/chat (POST)
- Data sources: PubMed API + bioRxiv API (real-time online access)
- Local cache: ChromaDB, /data/oih/knowledge/chroma, auto-accumulated
- Qwen tool name: rag_search

## Use Cases
- Understand target biology background before drug design
- Query clinical data and linker selection rationale for approved ADCs
- Find literature support for computational results
- Identify known binding pockets and key residues for a target

## API Invocation
```bash
# Search literature
curl -s --noproxy '*' "http://localhost:8080/api/v1/rag/search?query=HER2+ADC+trastuzumab&n_pubmed=5&n_biorxiv=2"

# RAG Q&A
curl -s --noproxy '*' -X POST http://localhost:8080/api/v1/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the known ADC drugs targeting HER2", "history": []}'
```

## Qwen Invocation Parameters
- query: search keywords (supports both Chinese and English)
- n_pubmed: number of PubMed results to return (default 5)
- n_biorxiv: number of bioRxiv results to return (default 2)

## Upstream/Downstream Dependencies
- Upstream: target/drug names from user natural language requests
- Downstream: search results serve as background knowledge, guiding rfdiffusion hotspot selection, linker_select parameters, and gnina docking parameters

## Notes
- GET request with parameters in query string
- No proxy required; PubMed/bioRxiv are accessed directly
- Local ChromaDB auto-accumulates; knowledge base grows with usage
- First search on a topic is slower (online); subsequent queries use local cache and are faster
