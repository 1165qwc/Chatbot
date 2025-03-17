import streamlit as st
import PyPDF2
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

# Set OpenAI API key (replace with your actual key)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] # use streamlit secrets.toml for deployment

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def create_qa_chain(text):
    """Creates a question-answering chain using LangChain."""
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)

    if not texts:
        raise ValueError("No text chunks generated. Check text splitter configuration.")
    
    embeddings = OpenAIEmbeddings()
    try:
    docsearch = FAISS.from_texts(texts, embeddings)
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise
    
    llm = OpenAI()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    return qa_chain

def main():
    """Main function to run the Streamlit app."""
    st.title("Chat with Your Document")

    uploaded_file = st.file_uploader("Upload a PDF or text document", type=["pdf", "txt"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == "txt":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file type. Please upload a PDF or TXT file.")
            return

        qa_chain = create_qa_chain(text)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the document"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                response = qa_chain.run(prompt)
                full_response = response
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
