import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup API Keys and Models
PINECONE_API_KEY = "pcsk_56mM9j_zFm19vgnxjYykWJhzpjGaP46BZN54vRCjZCzufyrFCDLmUQzyM1iHmHndQPfuJ"
index_name = "jarvis"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="llama3", temperature=0)

st.title("🤖 Jarvis: Personal AI Assistant")

# 2. Sidebar for Document Upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF for Jarvis to learn", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Sync with Pinecone
        vectorstore = PineconeVectorStore.from_documents(
            splits, embeddings, index_name=index_name, pinecone_api_key=PINECONE_API_KEY
        )
        st.success("Jarvis has learned the document!")

# 3. Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Jarvis anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval Chain
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "You are Jarvis, a helpful assistant. Use the following context to answer the question. "
        "If you don't know, say you don't know.\n\n{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": prompt})
    
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})