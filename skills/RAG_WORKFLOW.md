# RAG 文献检索工作流

## 基本信息
- 端点：/api/v1/rag/search（GET），/api/v1/rag/chat（POST）
- 数据源：PubMed API + bioRxiv API（实时联网）
- 本地缓存：ChromaDB，/data/oih/knowledge/chroma，自动沉淀
- Qwen工具名：rag_search

## 使用场景
- 药物设计前了解靶点生物学背景
- 查询已上市ADC的临床数据和linker选择依据
- 验证计算结果的文献支撑
- 了解靶点已知结合口袋和关键残基

## API调用
```bash
# 检索文献
curl -s --noproxy '*' "http://localhost:8080/api/v1/rag/search?query=HER2+ADC+trastuzumab&n_pubmed=5&n_biorxiv=2"

# RAG问答
curl -s --noproxy '*' -X POST http://localhost:8080/api/v1/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "HER2靶点的已知ADC药物有哪些", "history": []}'
```

## Qwen调用参数
- query: 检索关键词（支持中英文）
- n_pubmed: PubMed返回条数（默认5）
- n_biorxiv: bioRxiv返回条数（默认2）

## 上下游关系
- 上游：用户自然语言请求中的靶点/药物名称
- 下游：检索结果作为背景知识，指导rfdiffusion hotspot选择、linker_select参数、gnina对接参数

## 注意事项
- GET请求，参数在query string
- 无需代理，PubMed/bioRxiv直连
- 本地ChromaDB自动沉淀，知识库随使用增长
- 首次检索同一主题较慢（联网），第二次走本地缓存更快
