import os
import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
from getpass import getpass


st.title("Chat with document")

st.text("Enter API Key")

api_key = st.text_input("API Key:", type="password")
os.environ['GOOGLE_API_KEY'] = api_key

if api_key:
    st.success("API Key entered successfully!")


    uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')


    question = st.text_input("Enter your question")

    if uploaded_file is not None and question:
        
        save_path = "./uploaded_files"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        data = source.fit(file_path, dtype="pdf", chunk_size=1024, chunk_overlap=0)
        retriever = retrieve.auto_retriever(data,type="normal",top_k=3)
        pipeline = generator.Generate(question=question, retriever=retriever)
        response = pipeline.call()
        
        st.write(response)


st.caption("Upload a PDF document and enter a question to query information from the document.")
