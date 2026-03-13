import os
import numpy as np
import faiss
import streamlit as st
import ollama

from pypdf import PdfReader

PDF_FOLDER = "pdfs"
CHAT_MODEL = "qwen2.5:0.5b"
EMBED_MODEL = "all-minilm"


def split_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        start = max(0, end - overlap)

    return chunks


def load_pdfs(folder: str = PDF_FOLDER):
    chunks = []
    metadata = []

    if not os.path.exists(folder):
        raise FileNotFoundError(f"A pasta '{folder}' não existe.")

    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"Nenhum PDF encontrado em '{folder}'.")

    for filename in pdf_files:
        path = os.path.join(folder, filename)
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                page_chunks = split_text(text)

                for chunk in page_chunks:
                    chunks.append(chunk)
                    metadata.append({
                        "arquivo": filename,
                        "pagina": page_num
                    })

    if not chunks:
        raise ValueError("Nenhum texto útil foi extraído dos PDFs.")

    return chunks, metadata


def embed_text(text: str):
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


@st.cache_resource
def build_index():
    chunks, metadata = load_pdfs()

    vectors = []
    for chunk in chunks:
        vectors.append(embed_text(chunk))

    embeddings = np.array(vectors, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, chunks, metadata


def retrieve_context(question: str, index, chunks, metadata, k: int = 2):
    q_vec = np.array([embed_text(question)], dtype="float32")
    _, indices = index.search(q_vec, k)

    selected_chunks = []
    selected_meta = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            selected_chunks.append(chunks[idx])
            selected_meta.append(metadata[idx])

    return selected_chunks, selected_meta


def answer_question(question: str, index, chunks, metadata):
    selected_chunks, selected_meta = retrieve_context(
        question, index, chunks, metadata, k=2
    )

    context = "\n\n".join(selected_chunks)
    context = context[:2500]

    fontes_unicas = []
    vistos = set()

    for item in selected_meta:
        chave = (item["arquivo"], item["pagina"])
        if chave not in vistos:
            vistos.add(chave)
            fontes_unicas.append(item)

    prompt = f"""
Use apenas este contexto para responder.
Se não encontrar a resposta, diga isso claramente.

Contexto:
{context}

Pergunta:
{question}
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    resposta = response["message"]["content"]
    return resposta, fontes_unicas


st.set_page_config(page_title="Professor de Medicina com Ollama", layout="wide")
st.title("🧠 Professor de Medicina com Ollama")
st.write("Faça perguntas com base nos PDFs da pasta `pdfs`.")

try:
    with st.spinner("Lendo PDFs e criando índice..."):
        index, chunks, metadata = build_index()

    pergunta = st.text_input("Pergunte sobre medicina:")

    if pergunta:
        with st.spinner("Buscando resposta..."):
            resposta, fontes = answer_question(pergunta, index, chunks, metadata)

        st.subheader("Resposta")
        st.write(resposta)

        if fontes:
            st.subheader("Fontes usadas")
            for fonte in fontes:
                st.write(f"- {fonte['arquivo']} | página {fonte['pagina']}")

except Exception as e:
    st.error(f"Erro: {e}")
