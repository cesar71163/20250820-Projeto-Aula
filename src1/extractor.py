# extractor.py
# Define schema, prompt, chain e função de extração

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM

load_dotenv()  # lê OPENAI_API_KEY do .env

class Nota(BaseModel):
    emitente: str | None = None
    nome_razao_social: str | None = None
    cpf_cnpj: str | None = None
    endereco: str | None = None

parser = JsonOutputParser(pydantic_schema=Nota)

prompt = PromptTemplate.from_template(
    """Você extrai informações de Notas Fiscais brasileiras (NF-e/NFC-e).

Retorne SOMENTE JSON válido e SOMENTE com as chaves:
- emitente
- nome_razao_social
- cpf_cnpj
- endereco

Regras:
- Extraia exatamente como aparece na nota.
- Se não encontrar um campo, use null.
- Se houver múltiplos CPFs/CNPJs, prefira o do emitente.

{format_instructions}

Texto da nota fiscal:
{text}
"""
).partial(format_instructions=parser.get_format_instructions())

# Deixe o LLM determinístico para extração


#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=300)
llm = OllamaLLM(model="llama3:8b")


chain = prompt | llm | parser

def extract_fields(text: str) -> dict:
    """Recebe texto livre e devolve dict com os campos do schema Nota."""
    return chain.invoke({"text": text})