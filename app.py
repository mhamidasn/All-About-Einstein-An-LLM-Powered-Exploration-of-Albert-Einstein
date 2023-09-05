from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

def main():
    # load the secret (ex: key)
    load_dotenv()

    # build the UI
    st.set_page_config(page_title="Ask about Albert Einstein", page_icon="🔍")
    st.markdown("<h1 style='text-align: center;'>🔍 Ask About Albert Einstein 🔍</h1>", unsafe_allow_html=True)

    # from read the pdf untill creae the vecore store/knowledge bas
    pdf_text = reading_pdf_text()
    text_chunks = get_text_chunks(pdf_text)
    knowledge_base = get_knowledge_base(text_chunks)

    # user question
    user_question = st.text_input("Ask a question about Albert Einstein:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.9, "max_length":512})
        chain = load_qa_chain(llm, chain_type='stuff')
        response = chain.run(input_documents=docs, question=user_question) 
        st.write(response)

def reading_pdf_text(pdf_path="docs/Albert Einstein -- Britannica Encyclopedia.pdf"):
    pdf = PdfReader(pdf_path)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_knowledge_base(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return knowledge_base

if __name__ == "__main__":
    main()