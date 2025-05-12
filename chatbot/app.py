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

load_dotenv()

st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄")

def validate_api_key(api_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test connection"}]
        )
        return True
    except Exception as e:
        st.error(f"API Key Validation Error: {str(e)}")
        return False

def get_openai_api_key():
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    
    st.sidebar.header("🔑 OpenAI API Key")
    api_key_form = st.sidebar.form(key="api_key_form")
    
    with api_key_form:
        api_key = st.text_input(
            "Enter your OpenAI API Key", 
            type="password", 
            help="You can find your API key at https://platform.openai.com/account/api-keys"
        )
        
        validate_button = st.form_submit_button("Validate API Key")
        
        if validate_button and api_key:
            if validate_api_key(api_key):
                st.session_state.openai_api_key = api_key
                st.sidebar.success("API Key validated for this session!")
                return api_key
            else:
                st.sidebar.error("Invalid API Key. Please try again.")
                return None
    
    return None

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
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.2, 
            openai_api_key=api_key
        )
        
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
        
        retrieval_chain = (
            RunnableParallel({
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", "")
            })
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return retrieval_chain, retriever
    except Exception as e:
        st.error(f"Error creating retrieval chain: {e}")
        return None, None

def main():
    st.title("📄 PDF Chat Assistant")
    st.markdown("Upload a PDF document and start asking questions about its content.")

    session_keys = [
        ('processed_docs', None),
        ('chat_history', []),
        ('uploaded_file_name', None),
        ('retrieval_chain', None),
        ('retriever', None),
        ('source_documents', None)
    ]
    
    for key, default_value in session_keys:
        if key not in st.session_state:
            st.session_state[key] = default_value

    api_key = get_openai_api_key()

    if api_key:
        uploaded_file = st.file_uploader("Upload any PDF...", type="pdf")

        if uploaded_file and (not st.session_state.processed_docs or uploaded_file.name != st.session_state.uploaded_file_name):
            if not isinstance(st.session_state.chat_history, list):
                st.session_state.chat_history = []
            
            with st.spinner("Processing your PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                st.session_state.processed_docs = process_pdf(temp_path)
                st.session_state.uploaded_file_name = uploaded_file.name
                
                if st.session_state.processed_docs:
                    retrieval_chain_result = create_retrieval_chain(
                        st.session_state.processed_docs, 
                        api_key
                    )
                    
                    st.session_state.retrieval_chain = retrieval_chain_result[0]
                    st.session_state.retriever = retrieval_chain_result[1]
                    
                    os.unlink(temp_path)
                    
                    st.success(f"PDF processed successfully! {len(st.session_state.processed_docs)} chunks created.")
                else:
                    st.error("Failed to process the PDF.")

        if isinstance(st.session_state.chat_history, list):
            for message in st.session_state.chat_history:
                with st.chat_message(message.get("role", "user")):
                    st.markdown(message.get("content", ""))

        if st.session_state.processed_docs and st.session_state.retrieval_chain:
            user_query = st.chat_input("Ask a question about the document")
            
            if user_query:
                st.chat_message("user").markdown(user_query)
                
                if not isinstance(st.session_state.chat_history, list):
                    st.session_state.chat_history = []
                
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
        st.sidebar.info("""
        ### How to Use:
        1. Obtain an OpenAI API Key
        2. Enter the key in the sidebar
        3. Validate the key for this session
        4. Upload a PDF
        5. Start chatting with your document
        """)

st.sidebar.header("About")
st.sidebar.markdown("""
This PDF Chat Assistant allows you to:
- Upload any PDF document
- Ask questions about its content
- Receive AI-powered answers
- View source document references
""")

if __name__ == "__main__":
    main()