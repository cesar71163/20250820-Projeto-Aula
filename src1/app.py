# app.py
import streamlit as st
import pandas as pd
from ocr import pdf_bytes_to_text
from extractor import extract_fields
import fitz
import numpy as np



st.set_page_config(page_title="Extração de dados de NF-e / Recibo", layout="wide")

# --------------------------
# SIDEBAR (entrada e config)
# --------------------------
st.sidebar.title("Entrada e Configurações")

uploaded = st.sidebar.file_uploader("Envie um PDF", type=["pdf"])
dpi = st.sidebar.slider("DPI (rasterização)", 200, 600, 300, step=50)
max_pages = st.sidebar.number_input("Ler até N páginas (0 = todas)", min_value=0, value=1, step=1)
langs = st.sidebar.multiselect("Idiomas OCR", ["pt", "en", "es"], default=["pt", "en"])
usar_gpu = st.sidebar.checkbox("Usar GPU (EasyOCR)", value=False)
mostrar_texto = st.sidebar.checkbox("Mostrar texto OCR", value=False)
mostrar_json = st.sidebar.checkbox("Mostrar JSON", value=False)
processar = st.sidebar.button("Processar")

# --------------------------
# PAINEL CENTRAL (resultados)
# --------------------------
st.title("Extração de dados de NF-e / Recibo (PDF)")

if not uploaded:
    st.info("Envie um PDF pelo painel à esquerda.")
elif processar:
    try:
        # Lê bytes do PDF uma vez
        pdf_bytes = uploaded.read()

        # Renderiza 1ª página em imagem (para exibir no painel)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)  # usa o mesmo DPI do OCR para consistência
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

        # Executa OCR
        with st.spinner("Executando OCR…"):
            texto = pdf_bytes_to_text(
                pdf_bytes=pdf_bytes,
                dpi=dpi,
                max_pages=None if max_pages == 0 else max_pages,
                langs=tuple(langs) if langs else ("pt",),
                gpu=usar_gpu,
            )

        # Extrai campos com LangChain
        with st.spinner("Estruturando com LangChain…"):
            dados = extract_fields(texto)

        # Layout lado a lado: imagem (esq) e informações (dir)
        col_img, col_info = st.columns([1.2, 1.8], gap="large")

        with col_img:
            st.subheader("Pré-visualização")
            st.image(img_np, caption="Página 1 do PDF", use_container_width=True)
            if mostrar_texto:
                st.caption("Texto OCR (primeiras linhas)")
                st.text_area("Conteúdo extraído", value=texto[:4000], height=240)

        with col_info:
            st.subheader("Metadados")
            df_meta = pd.DataFrame([dados])
            st.table(df_meta)

            if mostrar_json:
                st.subheader("JSON")
                st.json(dados)

    except Exception as e:
        st.error(f"Erro ao processar: {e}")