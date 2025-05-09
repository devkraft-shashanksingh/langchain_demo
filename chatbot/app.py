import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


st.title("LangChain RAG DEMO")
uploaded_file = st.file_uploader("Upload any PDF for Retrieval", type="pdf")
query = st.text_input("Ask question based on the uploaded PDF")

def process_pdf(file):
    loader = PyPDFLoader(file.name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return text_splitter.split_documents(documents)

def get_rag_chain(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

if uploaded_file and query:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing PDF and querying..."):
        docs = process_pdf(uploaded_file)
        chain = get_rag_chain(docs)
        result = chain({"query": query})
        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Sources")
        for doc in result["source_documents"]:
            st.write(doc.metadata.get("source", "Unknown"))
