import os
import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
from beyondllm.llms import GeminiModel
from getpass import getpass
from dotenv import load_dotenv
load_dotenv()
st.title("Chat with document")
google_api_key=os.getenv("GOOGLE_API_KEY")
uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
question = st.text_input("Enter your question")
if uploaded_file is not None and question:
    save_path = "./uploaded_files"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        data = source.fit(file_path, dtype="pdf", chunk_size=1024, chunk_overlap=250)
        llm = GeminiModel(model_name="gemini-pro",google_api_key)
        retriever = retrieve.auto_retriever(data,type="normal",top_k=3)
        pipeline = generator.Generate(question=question, retriever=retriever,llm=llm)
        response = pipeline.call()
        
        st.write(response)


st.caption("Upload a PDF document and enter a question to query information from the document.")
