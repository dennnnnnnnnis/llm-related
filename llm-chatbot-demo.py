import streamlit as st
import time
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatLlamaCpp

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
CHROMA_PATH = "chroma"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def compile_stream(stream):
    output = {}
    for chunk in stream:
        for key in chunk:
            if key not in output:
                output[key] = chunk[key]
            else:
                output[key] += chunk[key]
    return output

def to_stream(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

@st.dialog("References")
def source_dialog(context):
    for idx, doc in enumerate(context):
        if "source" in doc.metadata:
            st.markdown(f"Source {idx + 1}: {doc.metadata['source']}")

# Side bar
st.sidebar.header("LLM Chatbot")
# this is currently not used
key = st.sidebar.text_input("Enter your OpenAI Key: ", type="password")
# model = st.sidebar.selectbox("Choose your OpenAI model", ["gpt-3.5-turbo", "llama3-chat"])
# max_tokens = st.sidebar.slider("max_tokens", 0, 2000, 1024)
# tmp = st.sidebar.slider("temperature", 0.0, 5.0, 1.0)

# Create a form for model settings to batch the changes
with st.sidebar.form("model_settings"):
    model = st.selectbox(
        "Choose your model", 
        ["gpt-3.5-turbo", "llama3-chat"],
        key="model_selector"
    )
    max_tokens = st.slider("max_tokens", 0, 2000, 1024, key="max_tokens_slider")
    tmp = st.slider("temperature", 0.0, 5.0, 1.0, key="temperature_slider")
    
    # Add a submit button to apply changes
    submitted = st.form_submit_button("Apply Changes")

# Update session state when form is submitted
if submitted:
    st.session_state["openai_model"] = model
    st.session_state["max_tokens"] = max_tokens
    st.session_state["temperature"] = tmp
    st.rerun()


# Preparation
embedding_func = OpenAIEmbeddings(api_key=API_KEY)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# template for rag
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

Context: {context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(prompt_template)

# Main page
st.title("LLM Chatbot Demo (With RAG) -- Splendor")

# Initialize default values if not in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = max_tokens
if "temperature" not in st.session_state:
    st.session_state["temperature"] = tmp

if st.session_state["openai_model"] == "gpt-3.5-turbo":
    st.sidebar.info("Using GPT-3.5 Turbo")
    llm = ChatOpenAI(
        model=st.session_state["openai_model"], 
        api_key=API_KEY, 
        temperature=st.session_state["temperature"], 
        max_tokens=st.session_state["max_tokens"]
    )
elif st.session_state["openai_model"] == "llama3-chat":
    st.sidebar.info("Using Llama model via localhost:8080")
    llm = ChatOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="dummy",  # Llamafile doesn't need a real key
        temperature=st.session_state["temperature"],
        max_tokens=st.session_state["max_tokens"],
        streaming=True,
        model="local-model"  # can be any string
    )
    # llm = ChatLlamaCpp(
    #     model_path="http://localhost:8080", 
    #     temperature=st.session_state["temperature"],
    #     max_tokens=st.session_state["max_tokens"],
    #     streaming=True,
    # )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?", "source": None}]

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "assistant" and message["source"] != None:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if st.button("🔗", key=f"Source {i}"):
                source_dialog(message["source"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask anything about Splendor!"):
    # currently not adding this warning
    # if not key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "source": None})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # modify the model behaviour
    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        # response = st.write_stream(stream)
        question = st.session_state.messages[-1]["content"]

        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )
        # stream = rag_chain.stream(question)
        # response = st.write_stream(stream)

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain)

        output = compile_stream(rag_chain_with_source.stream(question))
        # print(output)
        st.write_stream(to_stream(output["answer"]))

        # render the sources on the frontend as well
        if st.button("🔗", key="display_purpose"):
            source_dialog(output["context"])

    st.session_state.messages.append({"role": "assistant", "content": output["answer"], "source": output["context"]})
