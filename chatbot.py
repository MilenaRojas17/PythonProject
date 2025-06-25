import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import vector_stores
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

# Streamlit UI
st.header("My first Chatbot (Without API Key)")

with st.sidebar:
    st.title("Chatbot")
    file = st.file_uploader("Upload a PDF to start", type="pdf")

# Procesar PDF
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()


    # Separar texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Crear embeddings locales con sentence-transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_model = HuggingFaceEmbeddings()
    embeddings = embedding_model.embed_documents(chunks)


    # Crear vector store con FAISS
    vector_store = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    # Obtener pregunta del usuario
    user_question = st.text_input("Ask a question:")
    question_embedding = model.encode(user_question)

    if user_question:
        # Buscar similitudes

        question_embedding = np.array(question_embedding)
        if question_embedding.ndim == 1:
            question_embedding = np.array(question_embedding).reshape(1, -1)
            # Ensure it's a 2D array
            if isinstance(question_embedding, list):
                question_embedding = np.array(question_embedding)

                question_embedding = question_embedding.reshape(1, -1)
        docs_and_scores = vector_store.similarity_search_with_score(user_question, k=3)
        print("Query vector shape:", np.array(question_embedding).shape)
        assert isinstance(vector_store.index.d, object)
        print("Index dimensionality:", vector_store.index.d)

        # Generar respuesta (simplificada)
        st.subheader("Your answer:")
        for doc in docs_and_scores:
            st.write(doc[0].page_content)
