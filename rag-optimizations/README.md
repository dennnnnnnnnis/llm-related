# RAG Problems & Optimizations & Notes

This is for noting down anything related to RAG. Also act as a readme for optimizations or any varients of RAG implemented in this repository.

## List of Done

- `rag_recall.ipynb`: multi-retrieval (BM25 + vector embedding) techniques + Reciprocal Rank Fusion (RRF) for re-ranking
- `rag_video.ipynb`: RAG with videos as input
- `query_optimization.ipynb`: RAG with pre-retrieval related optimization techniques such as query rewrite, query expansion and query decompositions
- `contextual_compression.ipynb`: RAG with various techniques in contextual compression (retriever retrieves the documents, then compressor filter/reduce the documents): LLMChainExtractor, ChainFilter, Listwise Rerank, embedding filter and combining methods
- `rerank.ipynb`: RAG + Pointwise reranking with Qwen3 reranker
- `rag_sql_generate.ipynb`: RAG for SQL generation in handling database queries, with Chroma + Sqlite

## Q&A

**Q: Why multi-retrieval?**

A: Multi-retrieval allows you to combine results from different retrieval strategies or sources, increasing the chances of finding the most relevant information for a query.

**Q: Why re-ranking?**

A: Re-ranking helps sort retrieved documents by relevance, improving the quality of the final context passed to the LLM.

**Q: How to choose TextSplitter?**

A: 

**Q: How to choose embedding model?**

- Consider the language(s) of documents
- The size and speed of the model (Latency)
- The quality of semantic similarity between retrieved documents and query
- Reference: https://www.mongodb.com/developer/products/atlas/choose-embedding-model-rag/

**Q: How to optimize when loading documents?**

A:

**Q: Why contextual compression?**

A: One challenge with retrieval is that usually you don't know the specific queries your document storage system will face when you ingest data into the system. This means that the information most relevant to a query may be buried in a document with a lot of irrelevant text. Passing that full document through your application can lead to more expensive LLM calls and poorer responses.

**Q: Why query optimization?**

A: Improves retrieval accuracy and efficiency by rewriting, expanding, or decomposing queries.