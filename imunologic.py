import os
import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.title("🧠 Professor de Medicina IA")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Ler PDFs ----------

def load_pdfs(folder="pdfs"):

    textos = []

    for file in os.listdir(folder):

        if file.endswith(".pdf"):

            reader = PdfReader(os.path.join(folder, file))

            for page in reader.pages:

                txt = page.extract_text()

                if txt:
                    textos.append(txt)

    return textos


# ---------- Criar banco vetorial ----------

@st.cache_resource
def build_index():

    textos = load_pdfs()

    embeddings = model.encode(textos)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, textos


index, textos = build_index()

# ---------- Interface ----------

pergunta = st.text_input("Pergunte sobre medicina:")

if pergunta:

    q_embed = model.encode([pergunta])

    D, I = index.search(np.array(q_embed), k=3)

    contexto = ""

    for i in I[0]:
        contexto += textos[i] + "\n"

    prompt = f"""
    Use o conteúdo abaixo para responder.

    Conteúdo:
    {contexto}

    Pergunta:
    {pergunta}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    st.write(response.choices[0].message.content)
