import os
import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
from beyondllm.llms import GeminiModel
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

st.title("Chat with Document")

# Ensure the API key is loaded correctly
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
question = st.text_input("Enter your question")

if uploaded_file is not None and question:
    save_path = "./uploaded_files"
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(save_path, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Assuming 'pdf_path' should be 'file_path'
    data = source.fit(file_path, dtype="pdf", chunk_size=4028, chunk_overlap=500)

    llm = GeminiModel(model_name="gemini-pro", google_api_key=google_api_key)
    retriever = retrieve.auto_retriever(data, type="normal", top_k=3)
    pipeline = generator.Generate(question=question, retriever=retriever, llm=llm)
    response = pipeline.call()

    st.write(response)

st.caption("Upload a PDF document and enter a question to query information from the document.")
