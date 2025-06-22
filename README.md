# legalai-chatbot

This repository is for storing the chatbot (with RAG) that I built during the internship last year in the University of Melbourne, it is only for demonstration purpose and not the final version of the chatbot they were using (as it's not allowed to copy the codes into private repository like this directly).

As I was part of the legalai research team, I was involved in the second half section of the whole pipline of the project. I witnessed the resulting chatbot they built, and I took responsibility for the evaluation of the system, connecting LLMs to the systems, and adding additional functionalities to the system etc. I wasn't part of the team building the prototype of the chatbot. However, the `chatbot-demo` achieves a highly similar effect with the resulting chatbot they used for the project, it has bascially all the features of the chatbot they used, except for the reference dataset (RAG-use).

## Techniques Used

- Overall: LangChain for LLM framework + Streamlit as frontend
- Simple `RecursiveCharacterTextSplitter` to split the documents + Chroma as the vector database with OpenAI embeddings
- Streanlit: Ability to switch between Llama and OpenAI models, with reference rendering with the responses as well
- LangChain for building RAG chain (Reference: [here](https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/))

## Further Studies and RAG Optimizations

Please refer to `rag-optimizations` folder for more