import streamlit as st
from openai import OpenAI
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

st.set_page_config(page_title="ChatGPT Médico da Cami")

st.title("🩺 ChatGPT Médico da Cami")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Cami: pergunte algo sobre medicina...")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    st.chat_message("user").write(prompt)

    system_prompt = """
    Você é um assistente médico educacional.
    Responda com base em medicina baseada em evidências.
    Explique conceitos de forma clara para estudantes de medicina.
    """

    messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    try:

       response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=messages
       )

       reply = response.choices[0].message.content

       st.session_state.messages.append(
          {"role": "assistant", "content": reply}
       )

       st.chat_message("assistant").write(reply)

    except Exception as e:
        st.error("Erro ao acessar a API. Verifique limites ou créditos.")

 





