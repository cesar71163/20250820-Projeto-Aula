import streamlit as st
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate



from langchain_ollama import OllamaEmbeddings


st.set_page_config(page_title="Conversar com documento")
st.title("Conversa com Documento")


PROMPT_TXT = """Você é um assistente para leitura de ARTIGO/RELATÓRIO científico.
                Responda SEMPRE em português, com clareza e objetividade.
                Use EXCLUSIVAMENTE o CONTEXTO (trechos do documento). Não use conhecimento externo.
                Se a informação não estiver no contexto, responda: "Não encontrei essa informação no documento."

                === HISTÓRICO ===
                {chat_history}

                === CONTEXTO (trechos) ===
                {context}

                === PERGUNTA ===
                {question}

                RESPOSTA:"""


uploaded = st.sidebar.file_uploader("Envie um artigo científico em PDF", 
                                    type=["pdf"])
k = st.sidebar.slider("k (nº de trechos)", 2, 8, 4)
if st.sidebar.button("Limpar"):
    for key in ("db",'qa','memory','msgs'):
        st.session_state.pop(key,None)
    st.rerun()


if uploaded and 'db'not in st.session_state:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name
    
    docs = PyPDFLoader(pdf_path).load()

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                            chunk_overlap=200).split_documents(docs)
    
    emb = OllamaEmbeddings(model="nomic-embed-text:latest")
    st.session_state.db = FAISS.from_documents(chunks, emb)


# Monta cadeia conversacional quando houver índice
if "db" in st.session_state and ("qa" not in st.session_state):
    retriever = st.session_state.db.as_retriever(search_kwargs={"k": k})
    
    llm = OllamaLLM(model="llama3:8b", 
                    temperature=0.5)
   
    prompt = PromptTemplate.from_template(PROMPT_TXT)
    
    memory = ConversationBufferMemory(memory_key="chat_history", 
                                      return_messages=True)
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    st.session_state.qa = qa
    st.session_state.memory = memory
    st.session_state.msgs = []

# UI de chat
if "qa" not in st.session_state:
    st.info("Envie um PDF na barra lateral para iniciar.")
else:
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Pergunte algo do documento…")
    if q:
        st.session_state.msgs.append({"role": "user", "content": q})
        out = st.session_state.qa.invoke({"question": q})
        ans = out["answer"]
        st.session_state.msgs.append({"role": "assistant", "content": ans})

        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(ans)
