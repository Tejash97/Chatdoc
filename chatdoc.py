import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import aisuite as ai

# Load environment variables
load_dotenv()

# API Keys
groq_api_key = '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'

# Initialize AI clients
client = ai.Client()
llm = ChatGroq(
    temperature=0.4,
    groq_api_key=groq_api_key,
    model_name="llama-3.1-70b-versatile"
)

# Document Q&A prompt
document_prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Streamlit app title
st.title("Tejash Gpt")

# Sidebar for document upload
uploaded_file = st.sidebar.file_uploader("Upload PDF document for Q&A", type=["pdf"])

# Chat UI initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# Helper function for document embedding
def process_document(file):
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.getbuffer())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

# Function to handle general AI responses
def get_ai_response(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="groq:llama-3.2-3b-preview",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle document-based Q&A
def get_document_response(question):
    if st.session_state.vectors is None:
        return "Please upload a document first."
    document_chain = create_stuff_documents_chain(llm, document_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response["answer"]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle file upload with status feedback
if uploaded_file and st.session_state.vectors is None:
    with st.spinner("Processing document embeddings..."):
        try:
            process_document(uploaded_file)
            st.success("Document embeddings generated successfully!")
        except Exception as e:
            st.error(f"Error in processing document: {str(e)}")

# Chat input for user
if user_input := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Handle document or general chat Q&A
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.vectors and "document" in user_input.lower():
                response = get_document_response(user_input)
            else:
                response = get_ai_response(user_input)
            st.markdown(response)

    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
