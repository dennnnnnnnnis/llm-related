import os
import shutil
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = "chroma"
DATA_FILE = "data/ba-splendor-rulebook copy.pdf"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = PyPDFLoader(DATA_FILE)
    pages = loader.load_and_split()
    print(pages[0])
    return pages

def split_text(docs):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )
    chunks = text_spliter.split_documents(docs)
    print(len(chunks))
    return chunks

def save_to_chroma(chunks):
    # clear out the data when putting in new vector embeddings
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(api_key=st.secrets["openai"]["OPENAI_API_KEY"]),
        persist_directory=CHROMA_PATH
    )
    print("Vector embeddings saved successfully")


if __name__ == "__main__":
    main()