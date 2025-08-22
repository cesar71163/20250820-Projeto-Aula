import fitz  # PyMuPDF
import numpy as np
import easyocr

def get_reader(langs=('pt','en'), gpu=False):
    return easyocr.Reader(list(langs), gpu=gpu)

def pdf_bytes_to_text(pdf_bytes: bytes, 
                      dpi: int = 300, 
                      max_pages: int | None = None,
                      langs=('pt','en'), 
                      gpu=False,
                      prefer_native: bool = True,
                      min_chars_native: int = 50) -> str:
    """
    Lê um PDF (bytes) e retorna texto.
    - Se prefer_native=True: tenta extrair TEXTO NATIVO do PDF primeiro.
      Se houver texto suficiente (>= min_chars_native), retorna esse texto.
      Caso contrário, faz OCR (EasyOCR) página a página.
    - Se prefer_native=False: sempre faz OCR.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_idx = range(len(doc)) if max_pages is None else range(min(len(doc), max_pages))

    # tentar texto nativo
    if prefer_native:
        textos_nativos = []
        for i in pages_idx:
            t = doc[i].get_text().strip()  # "text" já é o mais direto
            if t:
                textos_nativos.append(t)
        texto_concat = " ".join(textos_nativos).strip()
        if len(texto_concat) >= min_chars_native:
            return " ".join(texto_concat.split())  # normaliza espaços

    # 2) fallback: OCR (quando é PDF escaneado/sem texto embutido)
    reader = get_reader(langs=langs, gpu=gpu)
    textos_ocr = []
    for i in pages_idx:
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

        results = reader.readtext(
            img,
            detail=1,
            paragraph=True,
            decoder='beamsearch',
            beamWidth=10,
            text_threshold=0.4,
            low_text=0.2,
            contrast_ths=0.05,
            adjust_contrast=0.7,
        )
        texto = " ".join(r[1] for r in results)
        textos_ocr.append(texto)

    return " ".join(" ".join(t.split()) for t in textos_ocr)  # normaliza espaços