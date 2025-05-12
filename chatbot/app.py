import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Chat Assistant for PDF")

st.title("Chat Assistant")
st.markdown("Upload a PDF document and start asking questions about its content.")

def validate_api_key(api_key):
    """
    Validate the OpenAI API key by attempting to create a ChatOpenAI instance
    """
    try:
        ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.2)
        return True
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        return False

def get_openai_api_key():
    """
    Retrieve or prompt for OpenAI API key
    """
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        st.warning("OpenAI API key not found in environment variables.")
        api_key = st.text_input(
            "Enter your OpenAI API Key", 
            type="password", 
            help="You can find your API key at https://platform.openai.com/account/api-keys"
        )
        
        if api_key:
            save_key_col, validate_col = st.columns(2)
            with save_key_col:
                if st.button("Save API Key"):
                    try:
                        with open('.env', 'a') as env_file:
                            env_file.write(f"\nOPENAI_API_KEY={api_key}")
                        st.success("API Key saved to .env file!")
                        load_dotenv(override=True)
                    except Exception as e:
                        st.error(f"Error saving API key: {e}")
            
            with validate_col:
                if st.button("Validate API Key"):
                    if validate_api_key(api_key):
                        st.success("API Key is valid!")
                    else:
                        st.error("Invalid API Key")
    
    return api_key

def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def create_retrieval_chain(docs, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=api_key)
    
    template = """You are an assistant for question-answering tasks related to the uploaded PDF document. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Previous Conversation:
    {chat_history}

    Question: {question}
    
    Context:
    {context}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def retrieve_docs(input_dict):
        question = input_dict.get("question", "")
        if isinstance(question, str):
            return retriever.invoke(question)
        else:
            raise ValueError(f"Expected string for question, got {type(question)}")
    
    retrieval_chain = (
        RunnableParallel({
            "context": lambda x: format_docs(retrieve_docs(x)),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", "")
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return retrieval_chain, retriever

if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'source_documents' not in st.session_state:
    st.session_state.source_documents = None

api_key = get_openai_api_key()

if api_key:
    uploaded_file = st.file_uploader("Upload any PDF...", type="pdf")

    if uploaded_file and (not st.session_state.processed_docs or uploaded_file.name != st.session_state.uploaded_file_name):
        st.session_state.chat_history = []
        
        with st.spinner("Processing your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            st.session_state.processed_docs = process_pdf(temp_path)
            st.session_state.uploaded_file_name = uploaded_file.name
            
            st.session_state.retrieval_chain, st.session_state.retriever = create_retrieval_chain(
                st.session_state.processed_docs, 
                api_key
            )
            
            os.unlink(temp_path)
            
            if st.session_state.processed_docs:
                st.success(f"PDF processed successfully! {len(st.session_state.processed_docs)} chunks created.")
            else:
                st.error("Failed to process the PDF.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.processed_docs and st.session_state.retrieval_chain:
        user_query = st.chat_input("Ask a question about the document")
        
        if user_query:
            st.chat_message("user").markdown(user_query)
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query
            })
            
            with st.spinner("Searching for answers..."):
                try:
                    formatted_history = "\n".join([
                        f"Human: {msg['content']}" if msg['role'] == 'user' else f"AI: {msg['content']}"
                        for msg in st.session_state.chat_history[:-1]
                    ])
                    
                    try:
                        st.session_state.source_documents = st.session_state.retriever.invoke(user_query)
                    except Exception as e:
                        st.error(f"Error retrieving documents: {e}")
                        st.session_state.source_documents = []
                    
                    try:
                        answer = st.session_state.retrieval_chain.invoke({
                            "question": user_query,
                            "chat_history": formatted_history
                        })
                    except Exception as e:
                        import traceback
                        st.error(f"Chain invocation error: {e}")
                        st.error(f"Trace: {traceback.format_exc()}")
                        answer = "Sorry, I encountered an error processing your question."
                    
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    with st.expander("View Sources", expanded=False):
                        for i, doc in enumerate(st.session_state.source_documents, 1):
                            st.markdown(f"**Source {i}**")
                            st.markdown(f"Page: {doc.metadata.get('page', 0) + 1}")
                            st.markdown("Content:")
                            st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                            st.divider()
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
    else:
        st.info("Please upload a PDF to start chatting")

else:
    st.error("Please provide a valid OpenAI API Key to use the application.")

st.sidebar.header("About")
st.sidebar.markdown("""
This app allows you to have a chat with a PDF document using AI.

1. Enter your OpenAI API Key
2. Upload a PDF document
3. Start asking questions
4. Get answers based on the document content
""")