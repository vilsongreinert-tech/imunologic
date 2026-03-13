import os
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.title("📚 Professor de Medicina com IA")

# Ler todos os PDFs
def load_pdfs(folder="pdfs"):
    text = ""
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content
    return text


# Dividir texto em pedaços
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# Criar banco vetorial
@st.cache_resource
def create_vector_db():

    text = load_pdfs()

    chunks = split_text(text)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

    db = FAISS.from_texts(chunks, embeddings)

    return db


db = create_vector_db()

pergunta = st.text_input("Pergunte algo sobre medicina:")

if pergunta:

    docs = db.similarity_search(pergunta, k=3)

    contexto = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Use o conteúdo abaixo para responder a pergunta.

    Conteúdo:
    {contexto}

    Pergunta:
    {pergunta}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    resposta = response.choices[0].message.content

    st.write(resposta)
    



