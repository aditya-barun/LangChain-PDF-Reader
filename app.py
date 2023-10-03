import streamlit as st
#import pickel
from PyPDF2 import PdfReader
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import huggingface_hub
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
#from langchain.callbacks import g
from dotenv import load_dotenv

def main():
    st.header('Chat with  PDF 💬')
    st.sidebar.title('Summa 1.0 By MarvelReturns')
    st.sidebar.markdown('''
    Team Members:
    - Alok
    - Neha
    - Sangeetha
    - Debojyoti
    - Aditya 
    ''')
    load_dotenv()

    # Upload a PDF File
    pdf = st.file_uploader("Upload your PDF File", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # spilit ito chuncks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embedding
        embeddings = HuggingFaceEmbeddings()

        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask Question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.2,
                                                                               "max_length": 500})
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)



if __name__ == '__main__':
    main()