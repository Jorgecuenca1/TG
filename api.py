import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import locale
import logging
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Carga del modelo Llama 2
def load_llama_model():
    try:
        logger.info("Cargando modelo Llama 2...")
        model_name = "TheBloke/Llama-2-13b-Chat-GPTQ"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        gen_cfg = GenerationConfig.from_pretrained(model_name)
        gen_cfg.max_new_tokens = 512
        gen_cfg.temperature = 0.0000001
        gen_cfg.return_full_text = True
        gen_cfg.do_sample = True
        gen_cfg.repetition_penalty = 1.11

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=gen_cfg
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Modelo cargado correctamente.")
        return llm
    except Exception as e:
        logger.error("Error cargando el modelo Llama 2: %s", str(e))
        raise

llm = load_llama_model()

# Inicialización de componentes
locale.getpreferredencoding = lambda: "UTF-8"
template = """
<s>[INST] <<SYS>>
You are an AI assistant. You are truthful, unbiased and honest in your response.

If you are unsure about an answer, truthfully say "I don't know"
<</SYS>>

{text} [/INST]
"""
prompt = PromptTemplate(input_variables=["text"], template=template)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
embeddings = HuggingFaceEmbeddings()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI LLM service"}

@app.post("/rag/web/")
def rag_from_web(request: QueryRequest):
    try:
        logger.info("Cargando documentos web...")
        web_loader = UnstructuredURLLoader(urls=["https://en.wikipedia.org/wiki/Solar_System"], mode="elements", strategy="fast")
        web_doc = web_loader.load()
        updated_web_doc = filter_complex_metadata(web_doc)

        chunked_web_doc = text_splitter.split_documents(updated_web_doc)
        db_web = FAISS.from_documents(chunked_web_doc, embeddings)

        prompt_template = """
<s>[INST] <<SYS>>
Use the following context to Answer the question at the end. Do not use any other information. If you can't find the relevant information in the context, just say you don't have enough information to answer the question. Don't try to make up an answer.

<</SYS>>

{context}

Question: {question} [/INST]
"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        Chain_web = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db_web.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
        )

        logger.info("Invocando RAG para la pregunta: %s", request.question)
        result = Chain_web.invoke(request.question)
        logger.info("RAG completado.")
        return {"response": result['result'].strip()}
    except Exception as e:
        logger.error("Error en RAG desde web: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
